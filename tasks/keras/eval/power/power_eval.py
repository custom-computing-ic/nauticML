import subprocess

from nautic import taskx

import shutil
import tempfile

from tasks.keras.eval.power.hls_builder import HLSBuilder
from tasks.keras.eval.power.tb_writer import TestbenchWriter

from pathlib import Path
import os

DEFAULT_TOP = "myproject"

VIVADO_TIMEOUT = 8 * 60 * 60 # 8 hour default timeout
TCL_SCRIPT = Path(__file__).parent / "tcl_scripts" / "full_power.tcl"
import keras

class KerasEnergy:
    @taskx
    def evaluate_energy(ctx, model):
        # model = keras.Sequential([
        #     keras.layers.Input(shape=(28, 28, 1)),
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(8, name='fc1'),
        #     keras.layers.Activation('relu', name='fc1_relu'),
        #     keras.layers.Dense(10, name='fc2'),
        #     keras.layers.Activation('softmax', name='fc2_softmax'),
        # ])

        # project_dir = tempfile.mkdtemp(prefix="hls4ml_power_")
        project_dir = "/mnt/ccnas2/bdp/gt922/tmp/nauticml_projects"
        Path(project_dir).mkdir(parents=True, exist_ok=True)

        hls_dir = os.path.join(project_dir, "nauticml_pe_prj")
        ctx.hls4ml.hls_project_dir = hls_dir

        try: 
            if shutil.which("vivado") is None:
                raise RuntimeError("Vivado not found on PATH — skipping power estimation")
            
            HLSBuilder.build_hls_from_model(ctx, model)

            if (code := KerasEnergy.run_vivado_power_estimation(ctx)) != 0:
                raise RuntimeError(f"Vivado power estimation failed with return code: {code}")
            
            dyn_power, resources = KerasEnergy.extract_power_and_resources(ctx)
            if dyn_power is None or resources is None:
                raise ValueError(f"dynamic power and resources not parsed from power estimation at {hls_dir}")

            # We only focus on dynamic power, as it's mostly what changes between designs 
            ctx.eval.energy = dyn_power
            # TODO: create artifact for the resources used

        except Exception as e:
            ctx.log.error(f"Power eval failed: {e}")
            HLSBuilder.dump_hls_failure_log(ctx.log, hls_dir)

            ctx.eval.energy = None

        finally:
            try:
                shutil.rmtree(project_dir, ignore_errors=True)
            except Exception as e:
                ctx.log.warning(
                    f"Failed to clean up project dir {project_dir}: {e}"
                )

    @staticmethod
    def extract_power_and_resources(ctx):
        output_dir = Path(ctx.hls4ml.hls_project_dir) / "power_estimation"
        power_xml = output_dir / "power_synth.xml"
        util_xml  = output_dir / "utilization.xml"

        dynamic_w = None
        for label, values in KerasEnergy._iter_rows(power_xml):
            if label == "Dynamic (W)" and values:
                try:
                    dynamic_w = float(values[0])
                except ValueError:
                    pass
                break

        util_keys = {
            "CLB LUTs*":      "lut",
            "CLB Registers":  "ff",
            "Block RAM Tile": "bram",
            "DSPs":           "dsp",
            "URAM":           "uram",
        }
        resources = {}
        for label, values in KerasEnergy._iter_rows(util_xml):
            if label in util_keys and values:
                try:
                    resources[util_keys[label]] = float(values[0])
                except ValueError:
                    pass

        return dynamic_w, resources

    @staticmethod
    def _iter_rows(xml_path):
        import xml.etree.ElementTree as ET

        if not xml_path.exists():
            return
        root = ET.parse(xml_path).getroot()
        for row in root.iter("tablerow"):
            cells = [c.get("contents", "").strip() for c in row.findall("tablecell")]
            if cells and cells[0]:
                yield cells[0], cells[1:]

    @staticmethod
    def run_vivado_power_estimation(ctx):
        log = ctx.log

        hls_dir = Path(ctx.hls4ml.hls_project_dir)

        verilog_dir = hls_dir / "myproject_prj" / "solution1" / "syn" / "verilog"
        tb_file     = hls_dir / "power_tb.v"
        output_dir  = hls_dir / "power_estimation"

        if not verilog_dir.exists():
            log.error(f"Verilog dir not found: {verilog_dir}")
            return -1
        
        if not tb_file.exists():
            log.error(f"TB not found: {tb_file}")
            return -1
        
        if not (hls_dir / "input_vectors.dat").exists():
            log.error(f"input_vectors.dat not found in {hls_dir}")
            return -1

        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(hls_dir / "input_vectors.dat", output_dir / "input_vectors.dat")

        cmd = [
            "vivado", "-mode", "batch",
            "-source", str(TCL_SCRIPT),
            "-tclargs",
            str(verilog_dir),
            str(tb_file),
            str(output_dir),
            ctx.hls4ml.hls_config.fpga_part,
            DEFAULT_TOP,
            str(TestbenchWriter.get_clock_period(ctx)),
        ]
        log.info(f"Running: {' '.join(cmd)}")

        timeout = ctx.hls4ml.vivado_timeout or VIVADO_TIMEOUT

        try:
            result = subprocess.run(
                cmd, capture_output=False, text=True,
                timeout=timeout, cwd=output_dir
            )
        except subprocess.TimeoutExpired:
            log.error(f"Vivado timed out after {timeout}s")
            return -1

        log.info(f"Completed Vivado power estimation")

        if result.returncode != 0:
            tail = "\n".join(((result.stdout or "") + (result.stderr or "")).splitlines()[-100:])
            log.error(f"Vivado failed (rc={result.returncode}):\n{tail}")
            return -1
        
        return 0