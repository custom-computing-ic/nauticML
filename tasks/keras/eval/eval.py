
from pathlib import Path

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
        ctx.eval.energy = KerasEval.evaluate_energy(ctx, model)
        # TODO: decouple these evaluations with a map of things to update and the acc function, and do the same in bayes opt for logging
        # TODO: decouple also the pareto frontier
        # TODO: add also latency to the pareto front generated
        
    def evaluate_energy(ctx, model) -> float | None:
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
                   
        import logging
        # silence hls4ml logs that aren't errors
        logging.getLogger('hls4ml').setLevel(logging.ERROR)
        
        import tempfile

        if shutil.which("vivado") is None:
            ctx.log.warning("Vivado not found on PATH — skipping power estimation")
            return None


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
        
        for layer in stripped_model.layers:
            if layer.name not in hls_config['LayerName']:
                continue

            rf = KerasEval.get_min_rf(layer, ctx.hls4ml.hls_fpga_part)

            # Floor RF for conv layers to kernel area. Fully-parallel conv
            # line buffers (RF=1) cause pathological HLS binding times —
            # binding hangs for hours because every mul/add gets its own
            # DSP and shift register in the sliding window.
            if isinstance(layer, tf.keras.layers.Conv2D):
                kh, kw = layer.kernel_size
                rf = max(rf, kh * kw)
            elif isinstance(layer, tf.keras.layers.Conv1D):
                rf = max(rf, layer.kernel_size[0])

            hls_config['LayerName'][layer.name]['ReuseFactor'] = rf

            # Pin accumulator/result precision to stop hls4ml auto-widening
            # (otherwise you get ap_fixed<19,...> which slows binding).
            layer_cfg = hls_config['LayerName'][layer.name]
            if isinstance(layer_cfg.get('Precision'), dict):
                layer_cfg['Precision']['accum']  = ctx.hls4ml.hls_precision
                layer_cfg['Precision']['result'] = ctx.hls4ml.hls_precision

        hls_config['Model']['Precision'] = ctx.hls4ml.hls_precision
        hls_config['Model']['Strategy']  = ctx.hls4ml.hls_strategy
        hls_config['Model']['ConvImplementation'] = 'LineBuffer'

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
            part=ctx.hls4ml.hls_fpga_part
        )
        
        # ensures that the bayesian model is synthesised as a deterministic one
        hls_model.config.config['Bayes'] = False

        # Write the HLS C++ project files, then run HLS synthesis to generate Verilog
        hls_model.write()
        
        # If the hls4ml fork still injects 'mask_index' into the C++, 
        # this snippet comments it out before the build runs.
        cpp_file = os.path.join(hls_dir, 'firmware', 'myproject.cpp')
        if os.path.exists(cpp_file):
            with open(cpp_file, 'r') as f:
                lines = f.readlines()
            with open(cpp_file, 'w') as f:
                for line in lines:
                    if 'mask_index' in line:
                        f.write(f"// {line}") # Comment out the ghost interface
                    else:
                        f.write(line)
        
        hls_model.build(csim=False, synth=True, cosim=False, export=False)

        # Store project path for downstream Vivado synthesis
        ctx.hls4ml.hls_project_dir = hls_dir

        power_est = KerasEval._run_vivado_power_estimation(ctx, hls_dir, project_dir)

        # in case of error for power estimation, skip the metric
        if power_est is None:
            return None
        
        power_util, timing = power_est

        # We only focus on returning the dynamic power consumption, that is
        # because the static power is mostly constant across fpga designs, and is influenced mostly by
        # the choice of FPGA and not the design itself. 
        return timing["t_total_sec"] * (power_util["power"]["dynamic"])

    @staticmethod
    def get_min_rf(layer, fpga_part):
        # True DSP usage is higher, but this approximation estimates a lower bound that is FPGA specific
        DSP_COUNT = {
            "xcku115-flvb2104-2-i": 5520,
            "xcu250-figd2104-2l-e": 12288,
        }
        
        weights = layer.get_weights()
        if not weights:
            return 1
        
        total = np.prod(weights[0].shape)
        # fallback to 4096 if unknown
        max_parallel = DSP_COUNT.get(fpga_part, 4096)  
        
        min_rf = max(1, int(np.ceil(total / max_parallel)))
        
        # round up to nearest valid divisor
        return next(i for i in range(min_rf, total + 1) if total % i == 0)

    @staticmethod
    def _parse_timing_latency(report_dir: Path, hls_dir: str, num_samples: int, target_period_ns: float):
        import xml.etree.ElementTree as ET
        import re

        # 1. Parse HLS Performance Estimates
        hls_report = Path(hls_dir) / "myproject_prj" / "solution1" / "syn" / "report" / "myproject_csynth.xml"
        latency = 0
        
        if hls_report.exists():
            root = ET.parse(hls_report).getroot()
            # Latency: Time for one sample to finish
            lat_node = root.find(".//SummaryOfOverallLatency/Average-caseLatency")
            latency = int(lat_node.text) if lat_node is not None else 0
            
            # Interval (II): Cycles between starting new samples
            int_min_node = root.find(".//SummaryOfOverallLatency/Interval-min")
            int_max_node = root.find(".//SummaryOfOverallLatency/Interval-max")

            if int_min_node is not None and int_max_node is not None:
                min_interval = int(int_min_node.text)
                max_interval = int(int_max_node.text)
            else:
                min_interval = latency
                max_interval = latency

        # 2. Get Actual Clock Period from Vivado
        timing_txt = report_dir / "timing_summary.txt"
        actual_period_ns = target_period_ns
        if timing_txt.exists():
            with open(timing_txt, 'r') as f:
                content = f.read()
                match = re.search(r'WNS\(ns\)\s+.*?\s+(-?\d+\.\d+)', content)
                if match:
                    wns = float(match.group(1))
                    if wns < 0:
                        actual_period_ns = target_period_ns - wns

        # Pipelined Design
        # Total Cycles = Latency of first sample + (remaining samples * II)
        if num_samples > 0:
            min_cycles = latency + (num_samples - 1) * min_interval
            max_cycles = latency + (num_samples - 1) * max_interval
        else:
            min_cycles = 0
            max_cycles = 0
            
        t_min = min_cycles * (actual_period_ns * 1e-9)
        t_max = max_cycles * (actual_period_ns * 1e-9)
        
        # we assume an average, as no variance is reported
        t_total = (t_min + t_max) / 2

        return {
            "latency_cycles": latency,
            "ii_min_cycles": min_interval,
            "ii_max_cycles": max_interval,
            "actual_period_ns": actual_period_ns,
            "t_total_sec": t_total,
            "t_min_sec": t_min,
            "t_max_sec": t_max,
        }
                    
    
    @staticmethod
    def _parse_power_utilization(report_dir):
        import xml.etree.ElementTree as ET
        
        def _rows(xml_path):
            if not xml_path.exists(): return
            root = ET.parse(xml_path).getroot()
            for row in root.iter("tablerow"):
                cells = [c.get("contents", "").strip() for c in row.findall("tablecell")]
                if cells and cells[0]:
                    yield cells[0], cells[1:]

        data = {"power": {}, "util": {}}
        
        # Parse Power
        power_xml = report_dir / "power_summary.xml"
        power_keys = {"Total On-Chip Power (W)": "total", "Dynamic (W)": "dynamic", "Device Static (W)": "static"}
        for label, values in _rows(power_xml):
            if label in power_keys and values:
                data["power"][power_keys[label]] = float(values[0])

        # Parse Utilization
        util_xml = report_dir / "utilization.xml"
        util_keys = {"CLB LUTs*": "lut", "DSPs": "dsp", "CLB Registers": "ff", "Block RAM Tile": "bram"}
        for label, values in _rows(util_xml):
            if label in util_keys and len(values) >= 1:
                data["util"][util_keys[label]] = float(values[0])
                
        return data

    @staticmethod
    def _run_vivado_power_estimation(ctx, hls_dir: str, output_dir: str) -> tuple[dict, dict] | None:
        import subprocess
        from pathlib import Path

        log = ctx.log

        VIVADO_TIMEOUT = 180 * 60
        original_dir = Path(__file__).resolve().parent

        tcl_script = original_dir / "vivado_power_estimation.tcl"
        verilog_dir = Path(hls_dir)  / "myproject_prj" / "solution1" / "syn" / "verilog"
        report_dir = Path(output_dir) / "power_report"
        
        try:                
            os.chdir(hls_dir)
            log.info(f"Running Vivado power estimation")

            cmd = f'vivado -mode batch -source "{tcl_script.absolute()}" -tclargs "{verilog_dir.absolute()}" "{report_dir.absolute()}" "{ctx.hls4ml.hls_fpga_part}" myproject'

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=VIVADO_TIMEOUT)
            returncode = result.returncode
            
            os.chdir(original_dir)
            if returncode == 0:
                log.info(f"Power estimation completed successfully")

                power_util = KerasEval._parse_power_utilization(report_dir)
                timing = KerasEval._parse_timing_latency(report_dir, hls_dir, ctx.hls4ml.num_mc_samples.get(), ctx.hls4ml.hls_clock_period)
                
                return power_util, timing
                
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

        # return nothing on an erroneous execution
        return None

    @staticmethod
    def _strip_for_hls(model):
        import tensorflow as tf
        from tensorflow_model_optimization.sparsity.keras import strip_pruning
        from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer

        # 1. Extract the inner model if wrapped (e.g., MonteCarloDropoutModel)
        source = model
        if hasattr(source, 'model') and isinstance(source.model, tf.keras.Model):
            source = source.model

        # 2. Safely strip pruning wrappers
        # This bakes the masks into the weights (zeros remain zeros)
        try:
            source = strip_pruning(source)
        except Exception:
            # If not a pruned model, continue with the source as-is
            pass

        # 3. Rebuild as a clean Sequential backbone
        clean = tf.keras.models.Sequential()
        input_shape = source.input_shape[1:] # Exclude batch dimension

        for i, layer in enumerate(source.layers):
            # Skip all forms of Dropout
            is_dropout = (
                isinstance(layer, (tf.keras.layers.Dropout, InferenceDropoutLayer)) or 
                'dropout' in layer.name.lower()
            )
            if is_dropout:
                continue

            # Get config and ensure the first layer has the correct input shape
            config = layer.get_config()
            if len(clean.layers) == 0:
                config['batch_input_shape'] = (None,) + input_shape
                
            new_layer = layer.__class__.from_config(config)
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