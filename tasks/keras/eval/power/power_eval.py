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

# Valid values for the hls4ml.proxy knob (see KerasEnergy.compute_proxy).
PROXY_MODES = ("ff", "params", "model_proxy", "dsp")


def _unwrap(val):
    """Unwrap a nautic context value, which may be wrapped in a ref."""
    return val.get() if hasattr(val, "get") else val


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
    def evaluate_energy(ctx, get_model):
        """Evaluate dynamic power / energy.

        ``get_model`` is a zero-arg callable that lazily loads the Keras model;
        it is only invoked on a cache miss, so a cached result avoids loading
        the checkpoint entirely.
        """
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

            # Resolve before creating anything so a config typo fails fast
            # without leaving an orphan project dir behind.
            proxy_mode = KerasEnergy._resolve_proxy_mode(ctx)

            project_root = "/mnt/ccnas2/bdp/gt922/tmp/nauticml_projects"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_dir = os.path.join(
                project_root,
                f"nauticml_pe_prj_iter{iter_num}_{timestamp}_pid{os.getpid()}",
            )
            Path(project_dir).mkdir(parents=True, exist_ok=True)
            ctx.hls4ml.hls_project_dir = project_dir

            try:
                # The Vivado estimator needs vivado on PATH; the proxy modes
                # only need the (cheaper) HLS C-synthesis below.
                if not proxy_mode and shutil.which("vivado") is None:
                    raise RuntimeError("Vivado not found on PATH — skipping power estimation")

                HLSBuilder.build_hls_from_model(ctx, get_model())

                max_ii = _unwrap(ctx.hls4ml.max_ii)

                if proxy_mode:
                    # Proxy mode: derive a cheap value from the synthesis report
                    # / model to stand in for power, then carry it through the
                    # same energy = power × max_ii × n_mc relationship.
                    proxy_value = KerasEnergy.compute_proxy(ctx, proxy_mode)
                    n_mc = int(_unwrap(ctx.hls4ml.num_mc_samples))
                    energy = proxy_value * max_ii * n_mc
                    ctx.log.info(
                        f"Power proxy [{proxy_mode}]={proxy_value}, max_ii={max_ii}, "
                        f"n_mc={n_mc}, energy={energy}"
                    )
                    power_metrics.update({
                        "power": proxy_value,
                        "energy": energy,
                        "max_ii": max_ii,
                        "resources": {},
                        "timing": {},
                    })
                else:
                    if (code := KerasEnergy.run_vivado_power_estimation(ctx)) != 0:
                        raise RuntimeError(f"Vivado power estimation failed with return code: {code}")

                    dyn_power, resources, timing = KerasEnergy.extract_power_and_resources(ctx)
                    if dyn_power is None or resources is None:
                        raise ValueError(f"dynamic power and resources not parsed at {project_dir}")

                    n_mc = int(_unwrap(ctx.hls4ml.num_mc_samples))
                    energy = dyn_power * max_ii * n_mc

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
    def _resolve_proxy_mode(ctx):
        """Return the configured power proxy mode, or None for estimator mode.

        Raises if the knob is set to an unknown value so a typo fails loudly
        rather than silently falling back to the Vivado estimator.
        """
        mode = _unwrap(getattr(ctx.hls4ml, "proxy", None))
        if mode is None or mode == "":
            return None
        if mode not in PROXY_MODES:
            raise ValueError(
                f"Unknown hls4ml.proxy={mode!r}; expected one of {PROXY_MODES}"
            )
        return mode

    @staticmethod
    def compute_proxy(ctx, mode):
        """Compute a cheap power/energy proxy after HLS C-synthesis.

        Modes:
          ff           -> pre_ff (HLS FF estimate)
          params       -> synthesized model parameter count
          model_proxy  -> pre_ff**2 * pre_interval_max**0.5
        """
        if mode == "params":
            params = _unwrap(ctx.hls4ml.model_params)
            if params is None:
                raise ValueError("proxy 'params': model_params not available on ctx")
            return float(params)

        pre_ff, pre_dsp, pre_interval_max = KerasEnergy.extract_csynth_estimates(ctx)

        if mode == "ff":
            if pre_ff is None:
                raise ValueError("proxy 'ff': could not read FF from csynth report")
            return float(pre_ff)
    
        if mode == "dsp":
            if pre_dsp is None:
                raise ValueError("proxy 'dsp': could not read DSP from csynth report")
            return float(pre_dsp)


        if mode == "model_proxy":
            if pre_ff is None or pre_interval_max is None:
                raise ValueError(
                    "proxy 'model_proxy': missing pre_ff/pre_interval_max in csynth report"
                )
            return (float(pre_ff)) ** 2 * float(pre_interval_max) ** 0.5

        raise ValueError(f"Unknown proxy mode: {mode!r}")

    @staticmethod
    def extract_csynth_estimates(ctx):
        """Read (pre_ff, pre_interval_max) from the top-level HLS C-synthesis report."""
        import xml.etree.ElementTree as ET

        report = (
            Path(ctx.hls4ml.hls_project_dir)
            / "myproject_prj" / "solution1" / "syn" / "report"
            / f"{DEFAULT_TOP}_csynth.xml"
        )
        if not report.exists():
            return None, None

        try:
            root = ET.parse(report).getroot()
        except ET.ParseError:
            return None, None

        def _txt(path):
            el = root.find(path)
            return el.text if el is not None and el.text else None

        ff = _txt(".//AreaEstimates/Resources/FF")
        dsp =  _txt(".//AreaEstimates/Resources/DSP")
        interval_max = _txt(".//PerformanceEstimates/SummaryOfOverallLatency/Interval-max")

        return (
            float(ff) if ff is not None else None,
            float(dsp) if dsp is not None else None,
            float(interval_max) if interval_max is not None else None,
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