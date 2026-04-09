
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from sklearn.metrics import accuracy_score
from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from nautic import taskx
from tasks.keras.trust.converter.dropout.mc_model import MonteCarloDropoutModel
import os

class KerasEval:

    @taskx
    def eval(ctx):

        co = {  "InferenceDropoutLayer": InferenceDropoutLayer,
            "MonteCarloDropoutModel": MonteCarloDropoutModel,
            "PruneLowMagnitude": pruning_wrapper.PruneLowMagnitude
        }

        model = load_model(ctx.experiment.ckpt_file, custom_objects=co)
        y_prob = model.predict(ctx.dataset.data["x_test"])

        ctx.eval.accuracy = KerasEval.evaluate_accuracy(ctx, y_prob)
        ctx.eval.ece = KerasEval.evaluate_ece(ctx, y_prob)
        ctx.eval.ape = KerasEval.evaluate_ape(ctx, model)
        ctx.eval.flops = KerasEval.evaluate_flops(ctx)
        ctx.eval.power = KerasEval.evaluate_power(ctx, model)

    def evaluate_power(ctx, model) -> float | None:
        """
        Estimate MCD power consumption by:
        1. Stripping non-synthesisable layers (dropout, pruning wrappers)
        to get a deterministic backbone suitable for hls4ml
        2. Converting to an hls4ml project in a temp directory
        3. (Future) Running Vivado synthesis + P&R to get single-pass power
        4. Multiplying by S (number of MC samples) to estimate full MCD power

        Note: Stripping dropout means we synthesize a single deterministic
        forward pass. The actual BayesCNN runs S stochastic passes reusing
        the same hardware, so total MCD power ≈ S × single_pass_power.
        See HEART 2025 (Que et al.), Table 5 — Vivado post-P&R power is
        reported for the deterministic backbone; MCD reuses it S times.
        """
        import shutil
        import hls4ml
        import tempfile

        if shutil.which("vivado") is None:
            ctx.log.warning("Vivado not found on PATH — skipping power estimation")
            return None

        # Number of MC forward passes (default from eval.mc_samples)
        num_mc_samples = ctx.hls4ml.num_mc_samples

        # --- Step 1: Strip non-synthesisable layers ---
        # Dropout and pruning wrappers have no HLS template in hls4ml.
        # Stripping preserves the trained (pruned) weights — zeros remain
        # zero, which hls4ml can exploit with the Resource strategy.
        stripped_model = KerasEval._strip_for_hls(model)

        # --- Step 2: Configure hls4ml conversion ---
        # Settings from ctx.hls4ml, following wa-hls4ml (Hawks et al.)
        # and HEART 2025 (Que et al.):
        #   - ap_fixed<16,6>: 16-bit fixed point, 6 integer bits (HEART 2025 §5.1)
        #   - Resource strategy with io_stream for CNNs (wa-hls4ml §2.1.1)
        #   - Kintex UltraScale KU115 at ~200MHz (HEART 2025 Table 5)
        hls_config = hls4ml.utils.config_from_keras_model(
            stripped_model, granularity='name'
        )
        hls_config['Model']['Precision'] = ctx.hls4ml.hls_precision
        hls_config['Model']['ReuseFactor'] = ctx.hls4ml.hls_reuse_factor
        hls_config['Model']['Strategy'] = ctx.hls4ml.hls_strategy

        # Generate hls4ml project in temp directory
        project_dir = tempfile.mkdtemp(prefix='hls4ml_power_')
        project_name = 'power_estimation_prj'
        hls_dir = os.path.join(project_dir, project_name)

        hls_model = hls4ml.converters.convert_from_keras_model(
            stripped_model,
            hls_config=hls_config,
            output_dir=hls_dir,
            io_type=ctx.hls4ml.hls_io_type,
            clock_period=ctx.hls4ml.hls_clock_period,
            part=ctx.hls4ml.hls_fpga_part,
        )

        # Write the HLS C++ project files, then run HLS synthesis to generate Verilog
        hls_model.write()
        hls_model.build(csim=False, synth=True, cosim=False, export=False)

        # Store project path for downstream Vivado synthesis
        ctx.hls4ml.hls_project_dir = hls_dir

        power_watts = KerasEval._run_vivado_power_estimation(ctx, hls_dir, project_dir)

        # in case of error for power estimation, skip the metric
        if power_watts < 0:
            return None

        # Full MCD power = S × single-pass power.
        return num_mc_samples * power_watts

    @staticmethod
    def _run_vivado_power_estimation(ctx, hls_dir: str, output_dir: str) -> float:
        import subprocess
        from pathlib import Path

        log = ctx.log

        VIVADO_TIMEOUT = 40 * 60 # 40 minutes
        original_dir = Path(os.getcwd())

        tcl_script = original_dir / "vivado_power_estimation.tcl"
        verilog_dir = Path(hls_dir)  / "myproject_prj" / "solution1" / "syn" / "verilog"
        report_dir = Path(output_dir) / "power_report"

        try:                
            os.chdir(hls_dir)
            log.info(f"Running Vivado power estimation")

            # Run command exactly as you would in terminal
            cmd = f'vivado -mode batch -source "{tcl_script.absolute()}" -tclargs "{verilog_dir.absolute()}" "{report_dir.absolute()}" myproject'

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=VIVADO_TIMEOUT)
            returncode = result.returncode
            
            # Go back to original directory
            os.chdir(original_dir)
            if returncode == 0:
                log.info(f"Power estimation completed successfully")

                def _rows(xml_path: Path):
                    import xml.etree.ElementTree as ET

                    """Yield (label, [values...]) for every tablerow in the report."""
                    root = ET.parse(xml_path).getroot()
                    for row in root.iter("tablerow"):
                        cells = [c.get("contents", "").strip() for c in row.findall("tablecell")]
                        if cells and cells[0]:
                            yield cells[0], cells[1:]

                POWER_XML = report_dir / "power_summary.xml"

                #Extract key power numbers (Watts) from a Vivado power report.
                power_keys = {
                    "Total On-Chip Power (W)": "total_power",
                    "Dynamic (W)":             "dynamic_power",
                    "Device Static (W)":       "static_power",
                    "Junction Temperature (C)": "junction_temp_c",
                    "Confidence Level":        "confidence",
                }

                power = {}
                for label, values in _rows(POWER_XML):
                    if label in power_keys and values:
                        v = values[0]
                        try:
                            power[power_keys[label]] = float(v)
                        except ValueError:
                            power[power_keys[label]] = v

                # UTILIZATION_XML = report_dir / "utilization.xml"
                # util_keys = {
                #     "CLB LUTs*":        "lut",
                #     "  LUT as Logic":   "lut_logic",
                #     "  LUT as Memory":  "lut_memory",
                #     "CLB Registers":    "ff",
                #     "Block RAM Tile":   "bram",
                #     "URAM":             "uram",
                #     "DSPs":             "dsp",
                #     "Bonded IOB":       "io",
                # }
                # out = {}
                # for label, values in _rows(UTILIZATION_XML):
                #     if label in util_keys and len(values) >= 5:
                #         used, _, _, available, util = values[:5]
                #         out[util_keys[label]] = {
                #             "used": float(used) if used else 0.0,
                #             "available": float(available) if available else 0.0,
                #             "util_pct": float(util.lstrip("<")) if util else 0.0,
                #         }

                return power["total_power"]
                
            else:
                TAIL_OUTPUT = 100

                all_output = (result.stdout or "") + "\n" + (result.stderr or "")
                last_100 = "\n".join(all_output.splitlines()[-TAIL_OUTPUT:])
                
                log.error(f"Vivado failed with return code {returncode}")
                log.error(f"Last {TAIL_OUTPUT} lines:\n{last_100}")
        
        except subprocess.TimeoutExpired:
            log.error(f"Vivado timed out after {VIVADO_TIMEOUT}s limit")
        
        except Exception as e:
            log.error(f"Error running Vivado: {e}")
        
        finally:
            # Make sure we're back in the original directory even if something fails
            os.chdir(original_dir)

        return -1.0

    def _strip_for_hls(model):
        """
        Return a clean Keras Sequential model suitable for hls4ml by:
        - Unwrapping PruneLowMagnitude wrappers (preserving pruned weights)
        - Removing InferenceDropoutLayer instances (no HLS template exists)

        The pruned weights (with zeros) are kept intact — hls4ml's Resource
        strategy can exploit the resulting sparsity.
        """
        import tensorflow as tf
        from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
        from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer

        # If wrapped in a custom model class (e.g. MonteCarloDropoutModel),
        # try to extract the inner Sequential
        source = model
        if hasattr(source, 'model') and isinstance(source.model, tf.keras.Model):
            source = source.model

        # Strip pruning wrappers if any layer is wrapped
        has_pruning = any(
            isinstance(layer, pruning_wrapper.PruneLowMagnitude)
            for layer in source.layers
        )
        if has_pruning:
            try:
                from tensorflow_model_optimization.sparsity.keras import strip_pruning
                source = strip_pruning(source)
            except Exception:
                # If strip_pruning fails, manually unwrap each layer below
                pass

        # Rebuild without dropout layers, manually unwrapping any
        # remaining PruneLowMagnitude wrappers
        clean = tf.keras.models.Sequential()
        first = True

        for layer in source.layers:
            # Skip dropout — no HLS template
            if isinstance(layer, InferenceDropoutLayer):
                continue
            if 'dropout' in layer.name.lower():
                continue

            # Manually unwrap PruneLowMagnitude if strip_pruning didn't catch it
            if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                # Extract the inner layer and its pruned weights
                inner_layer = layer.layer
                inner_config = inner_layer.get_config()

                if first:
                    if 'batch_input_shape' not in inner_config:
                        inner_config['batch_input_shape'] = layer.input_shape
                    first = False

                new_layer = inner_layer.__class__.from_config(inner_config)
                clean.add(new_layer)

                # Copy the pruned weights (with zeros baked in).
                # PruneLowMagnitude stores weights as [kernel, mask, threshold, ...]
                # The actual pruned kernel = kernel * mask
                pruned_weights = []
                for w in inner_layer.get_weights():
                    pruned_weights.append(w)
                # If the inner layer has fewer weights than what we got,
                # just take the first N matching the inner layer's weight shapes
                inner_weight_shapes = [w.shape for w in new_layer.get_weights()]
                final_weights = []
                available = list(layer.get_weights())
                for shape in inner_weight_shapes:
                    for i, w in enumerate(available):
                        if w.shape == shape:
                            final_weights.append(w)
                            available.pop(i)
                            break
                if final_weights:
                    new_layer.set_weights(final_weights)
            else:
                layer_config = layer.get_config()

                if first:
                    if 'batch_input_shape' not in layer_config:
                        layer_config['batch_input_shape'] = layer.input_shape
                    first = False

                new_layer = layer.__class__.from_config(layer_config)
                clean.add(new_layer)
                new_layer.set_weights(layer.get_weights())

        return clean

    def evaluate_ece(ctx, y_prob) -> float:
        y_logits    = np.log(y_prob/(1-y_prob + 1e-15))

        # we use CPU device as TF_DETERMINISTIC_OPS is not implemented in tf.math.bincount(x)
        with tf.device('/CPU:0'):
            ece_keras   = tfp.stats.expected_calibration_error(num_bins=ctx.eval.num_bins,
                logits=y_logits, labels_true=np.argmax(ctx.dataset.data["y_test"],axis=1), labels_predicted=np.argmax(y_prob,axis=1))
            
        return float(ece_keras)

    def evaluate_ape(ctx, model) -> float:
        def entropy(output):
            batch_size = output.shape[0]
            entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
            return entropy

        x = ctx.dataset.data["x_train"]

        # TODO: ask about mean - should be hard-coded?
        x_noise = np.random.normal(ctx.dataset.mean, ctx.dataset.std, size=x.shape).astype(x.dtype)

        return entropy(model.predict(np.ascontiguousarray(x_noise)))

    def evaluate_accuracy(ctx, y_prob):
        accuracy = float(accuracy_score(
            np.argmax(ctx.dataset.data["y_test"], axis=1),
            np.argmax(y_prob, axis=1)
        ))

        return accuracy

    def evaluate_flops(ctx):
        """
        Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
        Ignore operations used in only training mode such as Initialization.
        Use tf.profiler of tensorflow v1 api.
        """

        model = ctx.model.original

        batch_size = None
        if batch_size is None:
            batch_size = 1

        # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
        # FLOPS depends on batch size
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]

        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        return flops.total_float_ops