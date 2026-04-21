from nautic import taskx

import os
from pathlib import Path
import shutil    
import tempfile

import hls4ml 
import tensorflow as tf
import numpy as np

class KerasEnergy:

    @taskx
    def evaluate_energy(ctx, model):
        """
        Estimate MCD power consumption by:
        1. Stripping non-synthesisable layers (dropout, pruning wrappers)
        to get a deterministic backbone suitable for hls4ml
        2. Converting to an hls4ml project in a temp directory
        3. (Future) Running Vivado synthesis + P&R to get single-pass power
        4. Multiplying by S (number of MC samples) to estimate full MCD power

        Note: Stripping dropout means we synthesize a single deterministic
        forward pass. The actual BayesCNN runs S stochastic passes reusing
        the same hardware, so total MCD power ≈ S * single_pass_power.
        """

        if shutil.which("vivado") is None:
            ctx.log.warning("Vivado not found on PATH — skipping power estimation")
            ctx.eval.energy = None
            return

        project_dir = tempfile.mkdtemp(prefix='hls4ml_power_')
        hls_dir = os.path.join(project_dir, 'power_estimation_prj')

        ctx.hls4ml.hls_project_dir = hls_dir

        hls_config, stripped_model = KerasEnergy.convert_from_nauticml(ctx, model)

        hls_model = hls4ml.converters.convert_from_keras_model(
            stripped_model,
            hls_config=hls_config,
            output_dir=hls_dir,
            io_type=ctx.hls4ml.hls_config.io_type,
            clock_period=ctx.hls4ml.hls_config.clock_period,
            part=ctx.hls4ml.hls_config.fpga_part
        )

        hls_model = KerasEnergy.write_hls_model(ctx, hls_model)

        KerasEnergy.build_hls_model(ctx, hls_model)

        power_util, timing = KerasEnergy.run_vivado_power_estimation(ctx, project_dir)

        # in case of error for power estimation, skip the metric
        if power_util is None or timing is None:
            ctx.eval.energy = None
            return
        
        # We only focus on returning the dynamic power consumption, that is
        # because the static power is mostly constant across fpga designs, and is influenced mostly by
        # the choice of FPGA and not the design itself. 
        ctx.eval.energy = timing["t_total_sec"] * (power_util["power"]["dynamic"])

    # Convert model from NauticML into a model that can be synthesised by hls4ml - returning the model config 
    # and the converted model for transformation. Ensure that each layer has also adequate rf factor.
    @staticmethod
    def convert_from_nauticml(ctx, model):
        stripped_model = KerasEnergy.strip_for_hls(model)

        hls_config = hls4ml.utils.config_from_keras_model(
            stripped_model, granularity='name'
        )
        
        for layer in stripped_model.layers:
            if layer.name not in hls_config['LayerName']:
                continue

            rf = KerasEnergy.get_min_rf(layer, ctx.hls4ml.hls_config.fpga_part)

            hls_config['LayerName'][layer.name]['ReuseFactor'] = rf

            # Pin accumulator/result precision to stop hls4ml auto-widening
            layer_cfg = hls_config['LayerName'][layer.name]
            if isinstance(layer_cfg.get('Precision'), dict):
                layer_cfg['Precision']['accum']  = ctx.hls4ml.hls_config.precision
                layer_cfg['Precision']['result'] = ctx.hls4ml.hls_config.precision

        hls_config['Model']['Precision'] = ctx.hls4ml.hls_config.precision
        hls_config['Model']['Strategy']  = ctx.hls4ml.hls_config.strategy
        hls_config['Model']['ConvImplementation'] = 'LineBuffer'

        return hls_config, stripped_model

    # Strip a given model of all layers that cannot be synth by hls4ml
    @staticmethod
    def strip_for_hls(model):
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

        for layer in source.layers:
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

    # Get the minimum reuse factor for each layer - the lower the reuse factor, the longer the 
    # synthesis will take for hls4ml as more complex placing.
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
        max_parallel = DSP_COUNT.get(fpga_part, 4096)  
        
        min_rf = max(1, int(np.ceil(total / max_parallel)))
        
        # round up to nearest valid divisor
        rf = next(i for i in range(min_rf, total + 1) if total % i == 0)

        if isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            n_chan = layer.input_shape[-1]
            # Floor to kh*kw*n_chan so each output pixel serializes input channels
            rf = max(rf, kh * kw * n_chan)
        elif isinstance(layer, tf.keras.layers.Conv1D):
            rf = max(rf, layer.kernel_size[0])
        
        return rf

    # write the hls files for the model, making sure that any bayesian artifacts are removed
    # as we only make a single pass
    @staticmethod
    def write_hls_model(ctx, hls_model):
        hls_dir = ctx.hls4ml.hls_project_dir

        # ensures that the bayesian model is synthesised as a deterministic one, as we stripped all the bayesian layers
        hls_model.config.config['Bayes'] = False
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
        
        return hls_model

    @staticmethod
    def build_hls_model(ctx, hls_model):
        is_saif = ctx.hls4ml.power_mode.mode != "vectorless"

        # Feed vectors to the model for saif simulations
        if is_saif:
            hls_model.compile()

            n_saif_samples = ctx.hls4ml.power_mode.num_samples
            x_sample = ctx.dataset.data["x_test"][:n_saif_samples]
            x_sample = np.ascontiguousarray(x_sample.astype(np.float32))

            # This writes tb_data/tb_input_features.dat as a side effect
            _ = hls_model.predict(x_sample)

            ctx.log.info(f"Wrote {n_saif_samples} test vectors for SAIF stimulus")
        
        # Build WITH cosim so the TB and wrappers get generated is saif
        hls_model.build(csim=False, synth=True, cosim=is_saif, export=False)

    @staticmethod
    def run_vivado_power_estimation(ctx, output_dir: str):
        import subprocess
        from pathlib import Path

        log = ctx.log
        hls_dir = ctx.hls4ml.hls_project_dir

        VIVADO_TIMEOUT = 180 * 60
        original_dir = Path(__file__).resolve().parent

        tcl_script = original_dir / "vivado_power_estimation.tcl"
        verilog_dir = Path(hls_dir)  / "myproject_prj" / "solution1" / "syn" / "verilog"
        report_dir = Path(output_dir) / "power_report"

        power_mode = ctx.hls4ml.power_mode  # "vectorless" | "saif_synth" | "saif_impl"

        # only needed for SAIF modes
        tb_file = None
        if power_mode != "vectorless":
            tb_file = Path(hls_dir) / "myproject_prj" / "solution1" / "sim" / "verilog" / "myproject_test.v"
            if not tb_file.exists():
                log.error(f"Testbench not found at {tb_file} - cosim may have failed previously.")
                return None, None
            
        try:                
            os.chdir(hls_dir)
            log.info(f"Running Vivado power estimation in mode: {power_mode}")

            tclargs = (
                f'"{verilog_dir.absolute()}" '
                f'"{report_dir.absolute()}" '
                f'"{ctx.hls4ml.hls_fpga_part}" '
                f'myproject '
                f'{power_mode}'
            )
            if tb_file is not None:
                tclargs += f' "{tb_file.absolute()}"'

            cmd = f'vivado -mode batch -source "{tcl_script.absolute()}" -tclargs {tclargs}'

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=VIVADO_TIMEOUT)
            returncode = result.returncode
            
            os.chdir(original_dir)
            if returncode == 0:
                log.info(f"Power estimation completed successfully")

                power_util = KerasEnergy.parse_power_utilization(report_dir)
                timing = KerasEnergy.parse_timing_latency(
                    report_dir, hls_dir,
                    ctx.hls4ml.num_mc_samples.get(),
                    ctx.hls4ml.hls_clock_period
                )

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
        return None, None

    @staticmethod
    def parse_timing_latency(report_dir: Path, hls_dir: str, num_samples: int, target_period_ns: float):
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
    def parse_power_utilization(report_dir):
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