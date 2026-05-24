import subprocess

from nautic import taskx

import shutil
import tempfile
from datetime import datetime

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
        
        # Each BO iteration gets its own iter+timestamp project_dir to avoid
        # collisions between concurrent runs and stale-snapshot reuse by xsim.
        # Iter number first so dirs sort by BO iteration even if timestamps
        # collide (and so it's obvious which run produced a given dir).
        project_root = "/mnt/ccnas2/bdp/gt922/tmp/nauticml_projects"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            iter_num = ctx.bayes_opt.iteration.get()
        except Exception:
            iter_num = "manual"
        project_dir = os.path.join(
            project_root, f"nauticml_pe_prj_iter{iter_num}_{timestamp}"
        )
        Path(project_dir).mkdir(parents=True, exist_ok=True)

        hls_dir = project_dir
        ctx.hls4ml.hls_project_dir = hls_dir

        try:
            if shutil.which("vivado") is None:
                raise RuntimeError("Vivado not found on PATH — skipping power estimation")

            HLSBuilder.build_hls_from_model(ctx, model)

            if (code := KerasEnergy.run_vivado_power_estimation(ctx)) != 0:
                raise RuntimeError(f"Vivado power estimation failed with return code: {code}")

            dyn_power, resources, timing = KerasEnergy.extract_power_and_resources(ctx)
            if dyn_power is None or resources is None:
                raise ValueError(f"dynamic power and resources not parsed from power estimation at {hls_dir}")

            # power: Vivado-reported dynamic dynamic power (W).
            # energy: dyn_power × max_layer_II × num_mc_samples.
            #   Proportional to actual joules per prediction (one prediction
            #   = N MC forward passes). At fixed clock_period the missing
            #   1/f_clk factor is a constant — doesn't affect BO ranking but
            #   makes the metric's order of magnitude reflect "per prediction".
            max_ii_raw = ctx.hls4ml.max_ii
            max_ii = max_ii_raw.get() if hasattr(max_ii_raw, "get") else max_ii_raw
            n_mc_raw = ctx.hls4ml.num_mc_samples
            n_mc = n_mc_raw.get() if hasattr(n_mc_raw, "get") else n_mc_raw
            energy = dyn_power * max_ii * int(n_mc)

            ctx.eval.power = dyn_power
            ctx.eval.energy = energy

            ctx.log.info(
                f"Power eval: power={dyn_power:.4f} W, max_ii={max_ii} cycles, "
                f"n_mc={n_mc}, energy={energy:.2f} W·cycles"
            )

            KerasEnergy.save_power_artifacts(
                ctx, dyn_power=dyn_power, max_ii=max_ii,
                energy=energy, resources=resources, timing=timing,
            )

        except Exception as e:
            ctx.log.error(f"Power eval failed: {e}")
            HLSBuilder.dump_hls_failure_log(ctx.log, hls_dir)

            ctx.eval.power = None
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
        power_xml   = output_dir / "power_synth.xml"
        util_xml    = output_dir / "utilization.xml"
        timing_txt  = output_dir / "timing_summary.txt"

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

        timing = KerasEnergy._parse_timing(timing_txt)

        return dynamic_w, resources, timing

    @staticmethod
    def _parse_timing(txt_path):
        """Pull WNS / WHS from report_timing_summary's plain-text output."""
        if not txt_path.exists():
            return {}
        import re
        text = txt_path.read_text()
        out = {}
        m = re.search(r"WNS\s*\(ns\)\s*\|?\s*(-?\d+\.\d+)", text)
        if m:
            out["wns_ns"] = float(m.group(1))
        m = re.search(r"WHS\s*\(ns\)\s*\|?\s*(-?\d+\.\d+)", text)
        if m:
            out["whs_ns"] = float(m.group(1))
        return out

    @staticmethod
    def save_power_artifacts(ctx, *, dyn_power, max_ii, energy, resources, timing):
        """
        Persist post-synth power/util/timing reports for this BO iteration.
        project_dir is rm-tree'd in `finally`, so we copy what's worth keeping
        into experiment.save_dir/power_artifacts/iter_<N>/ and log table
        artifacts so they show up in the Prefect UI.
        """
        try:
            iter_num = ctx.bayes_opt.iteration.get()
        except Exception:
            iter_num = "manual"

        save_dir_raw = ctx.experiment.save_dir
        save_dir = Path(save_dir_raw.get() if hasattr(save_dir_raw, "get") else save_dir_raw)
        artifact_dir = save_dir / "power_artifacts" / f"iter_{iter_num}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        src_dir = Path(ctx.hls4ml.hls_project_dir) / "power_estimation"
        for fname in (
            "power_synth.xml",
            "power_synth_verbose.txt",
            "power_synth_hierarchical.txt",
            "utilization.xml",
            "utilization_hierarchical.txt",
            "timing_summary.txt",
            "switching_synth_breakdown.txt",
        ):
            src = src_dir / fname
            if src.exists():
                shutil.copy(src, artifact_dir / fname)

        ctx.log.artifact(
            key=f"power-iter-{iter_num}",
            table=[
                {"metric": "dynamic_w",        "value": dyn_power},
                {"metric": "max_ii_cycles",    "value": max_ii},
                {"metric": "energy_w_cycles",  "value": energy},
            ],
        )
        if resources:
            ctx.log.artifact(
                key=f"resources-iter-{iter_num}",
                table=[{"resource": k, "count": v} for k, v in resources.items()],
            )
        if timing:
            ctx.log.artifact(
                key=f"timing-iter-{iter_num}",
                table=[{"metric": k, "ns": v} for k, v in timing.items()],
            )

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