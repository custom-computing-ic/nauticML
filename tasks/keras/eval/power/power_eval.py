import subprocess

from nautic import taskx

import shutil
from datetime import datetime

from tasks.keras.eval.power.hls_builder import HLSBuilder
from tasks.keras.eval.power.power_cache import PowerCache
from tasks.keras.eval.power.tb_writer import TestbenchWriter

from pathlib import Path
import os

DEFAULT_TOP = "myproject"

VIVADO_TIMEOUT = 8 * 60 * 60 # 8 hour default timeout
TCL_SCRIPT = Path(__file__).parent / "tcl_scripts" / "full_power.tcl"

class KerasEnergy:

    @staticmethod
    def _check_strategy_constraints(ctx):
        try:
            strat_raw = ctx.bayes_opt.curr_strategy
            strat = strat_raw.get() if hasattr(strat_raw, "get") else strat_raw
        except Exception:
            ctx.log.warning("No curr_strategy on ctx; skipping constraint check")
            return True

        checkable = ("accuracy", "ece", "ape", "flops")

        for name in checkable:
            try:
                metric_raw = getattr(ctx.eval, name)
                metric_val = metric_raw.get() if hasattr(metric_raw, "get") else metric_raw
            except Exception:
                continue
            if metric_val is None:
                continue

            try:
                bounds_raw = getattr(strat, name, None)
                bounds = bounds_raw.get() if hasattr(bounds_raw, "get") else bounds_raw
            except Exception:
                continue
            if bounds is None:
                continue

            def _read(field):
                try:
                    v = bounds.get(field) if hasattr(bounds, "get") and not callable(getattr(bounds, "get", None).__self__.__class__ if False else None) else getattr(bounds, field, None)
                except Exception:
                    v = None

                try:
                    if hasattr(bounds, field):
                        raw = getattr(bounds, field)
                        return raw.get() if hasattr(raw, "get") else raw
                except Exception:
                    pass
                try:
                    raw = bounds[field]
                    return raw.get() if hasattr(raw, "get") else raw
                except Exception:
                    return None

            lo = _read("min")
            hi = _read("max")

            if lo is not None and metric_val < float(lo):
                ctx.log.info(
                    f"Constraint violated: {name}={metric_val} < min={lo} "
                    f"(strategy={getattr(strat, 'name', '?')}); skipping power eval"
                )
                return False
            if hi is not None and metric_val > float(hi):
                ctx.log.info(
                    f"Constraint violated: {name}={metric_val} > max={hi} "
                    f"(strategy={getattr(strat, 'name', '?')}); skipping power eval"
                )
                return False

        return True

    @taskx
    def evaluate_energy(ctx, model):
        try:
            iter_raw = ctx.bayes_opt.iteration
            iter_num = iter_raw.get() if hasattr(iter_raw, "get") else iter_raw
        except Exception:
            iter_num = "manual"

        power_metrics = {
            "power": None, "energy": None, "max_ii": None,
            "resources": {}, "timing": {},
        }

        project_dir = None
        from_cache = False

        if (cached_values := PowerCache.probe_cache(ctx)) is not None:
            power_metrics.update(cached_values)
            from_cache = True
            ctx.log.info(
                f"Using cached values for iter={iter_num}: "
                f"power={cached_values['power']} W, "
                f"energy={cached_values['energy']} W·cycles"
            )
        else:
            if not KerasEnergy._check_strategy_constraints(ctx):
                ctx.eval.power = None
                ctx.eval.energy = None
               
                return

            project_root = "/mnt/ccnas2/bdp/gt922/tmp/nauticml_projects"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_dir = os.path.join(
                project_root,
                f"nauticml_pe_prj_iter{iter_num}_{timestamp}_pid{os.getpid()}",
            )
            Path(project_dir).mkdir(parents=True, exist_ok=True)
            ctx.hls4ml.hls_project_dir = project_dir

            try:
                if shutil.which("vivado") is None:
                    raise RuntimeError("Vivado not found on PATH — skipping power estimation")

                HLSBuilder.build_hls_from_model(ctx, model)

                if (code := KerasEnergy.run_vivado_power_estimation(ctx)) != 0:
                    raise RuntimeError(f"Vivado power estimation failed with return code: {code}")

                dyn_power, resources, timing = KerasEnergy.extract_power_and_resources(ctx)
                if dyn_power is None or resources is None:
                    raise ValueError(f"dynamic power and resources not parsed at {project_dir}")

                max_ii_raw = ctx.hls4ml.max_ii
                max_ii = max_ii_raw.get() if hasattr(max_ii_raw, "get") else max_ii_raw
                n_mc_raw = ctx.hls4ml.num_mc_samples
                n_mc = n_mc_raw.get() if hasattr(n_mc_raw, "get") else n_mc_raw
                energy = dyn_power * max_ii * int(n_mc)

                ctx.log.info(
                    f"Power eval: power={dyn_power:.4f} W, max_ii={max_ii} cycles, "
                    f"n_mc={n_mc}, energy={energy:.2f} W·cycles"
                )

                power_metrics.update({
                    "power": dyn_power,
                    "energy": energy,
                    "max_ii": max_ii,
                    "resources": resources,
                    "timing": timing,
                })

            except Exception as e:
                ctx.log.error(f"Power eval failed: {e}")
                HLSBuilder.dump_hls_failure_log(ctx.log, project_dir)

            finally:
                try:
                    PowerCache.write_cache(ctx, **power_metrics)
                except Exception as e:
                    ctx.log.warning(f"Failed to write power cache: {e}")

                if project_dir is not None:
                    try:
                        shutil.rmtree(project_dir, ignore_errors=True)
                    except Exception as e:
                        ctx.log.warning(f"Failed to clean up {project_dir}: {e}")

        ctx.eval.power  = power_metrics["power"]
        ctx.eval.energy = power_metrics["energy"]

        KerasEnergy.save_power_artifacts(
            ctx,
            dyn_power=power_metrics["power"],
            max_ii=power_metrics["max_ii"],
            energy=power_metrics["energy"],
            resources=power_metrics["resources"],
            timing=power_metrics["timing"],
            copy_files=not from_cache,  
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
    def save_power_artifacts(ctx, *, dyn_power, max_ii, energy, resources, timing, copy_files=True):
        try:
            iter_raw = ctx.bayes_opt.iteration
            iter_num = iter_raw.get() if hasattr(iter_raw, "get") else iter_raw
        except Exception:
            iter_num = "manual"

        artifact_key_suffix = f"iter{iter_num}-pid{os.getpid()}"

        save_dir_raw = ctx.experiment.save_dir
        save_dir = Path(save_dir_raw.get() if hasattr(save_dir_raw, "get") else save_dir_raw)
        artifact_dir = save_dir / "power_artifacts" / f"iter_{iter_num}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        if copy_files:
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
            key=f"power-{artifact_key_suffix}",
            table=[
                {"metric": "dynamic_w",       "value": dyn_power},
                {"metric": "max_ii_cycles",   "value": max_ii},
                {"metric": "energy_w_cycles", "value": energy},
            ],
        )

        if resources:
            ctx.log.artifact(
                key=f"resources-{artifact_key_suffix}",
                table=[{"resource": k, "count": v} for k, v in resources.items()],
            )

        if timing:
            ctx.log.artifact(
                key=f"timing-{artifact_key_suffix}",
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