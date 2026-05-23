import hls4ml

import tensorflow as tf
import numpy as np

import re
import os

from pathlib import Path

from tasks.keras.eval.power.tb_writer import TestbenchWriter

class HLSBuilder:

    def build_hls_from_model(ctx, model):
        hls_dir = ctx.hls4ml.hls_project_dir

        # 1. Strip + build hls_config (NauticML-specific, retained from the
        #    original implementation but with safety passes from the SAIF code).
        hls_config, stripped_model = HLSBuilder.convert_from_nauticml(ctx, model)

        # 2. Pick io_type once; downstream TB writing branches on it.
        io_type = ctx.hls4ml.hls_config.io_type or "io_parallel"
        
        # Ensure io_stream for conv models
        has_conv = any(
            isinstance(l, (tf.keras.layers.Conv1D, tf.keras.layers.Conv2D))
            for l in stripped_model.layers
        )
        if has_conv and io_type != "io_stream":
            ctx.log.warning(
                f"Model has Conv layers but io_type={io_type!r}; forcing io_stream."
            )
            io_type = "io_stream"

        is_saif = ctx.hls4ml.power_mode.mode != "vectorless"
        input_tb_path = output_tb_path = None
        beats_per_sample = 1

        if is_saif:
            (
                input_tb_path,
                output_tb_path,
                _,
                _,
                beats_per_sample,
            ) = TestbenchWriter.load_saif_testcases(
                ctx, stripped_model, hls_dir, hls_config, io_type
            )

        hls_model = hls4ml.converters.convert_from_keras_model(
            stripped_model,
            hls_config=hls_config,
            output_dir=hls_dir,
            io_type=io_type,
            clock_period=ctx.hls4ml.hls_config.clock_period,
            part=ctx.hls4ml.hls_config.fpga_part,
            input_data_tb=input_tb_path,
            output_data_tb=output_tb_path,
            backend="Vitis",
        )

        # hls_model = HLSBuilder.write_hls_model_bayes(ctx, hls_model)
        hls_model.write()

        # Optional Vitis 2022.2+ TCL patch.
        if getattr(ctx.hls4ml, "patch_vitis_2022_2", False):
            HLSBuilder.patch_build_prj(Path(hls_dir), ctx.log)

        hls_model.build(csim=False, synth=True, cosim=False, export=False)

        if is_saif:
            TestbenchWriter.write_power_testbench(
                ctx,
                project_dir=Path(hls_dir),
                stripped_model=stripped_model,
                hls_config=hls_config,
                io_type=io_type,
                beats_per_sample=beats_per_sample,
                n_samples=TestbenchWriter._n_saif_samples(ctx),
            )

        return hls_model

    # ------------------------------------------------------------------
    # Model + hls_config preparation
    # ------------------------------------------------------------------

    @staticmethod
    def convert_from_nauticml(ctx, model):
        """
        Strip the NauticML model, then build an hls4ml config with per-layer
        reuse factors and pinned precision. Adds the two safety passes from
        the SAIF code (strategy-case normalisation, narrow-precision widening)
        and a guardrail against silent fallback to Model defaults.
        """
        stripped_model = HLSBuilder.strip_for_hls(model)

        hls_config = hls4ml.utils.config_from_keras_model(
            stripped_model, granularity="name"
        )

        # Compute each layer's natural min RF (existing per-layer logic), then
        # pin only the worst layer(s) — those whose natural min_rf equals the
        # max across the model — to the configured target. max(RF) stays
        # constant between DSE iterations (the bound is the config value, not
        # the architecture); smaller layers keep their natural RF and stay
        # efficient.
        target_rf = ctx.hls4ml.hls_config.reuse_factor
        fpga_part = ctx.hls4ml.hls_config.fpga_part

        natural_rfs = {
            layer.name: HLSBuilder.get_min_rf(layer, fpga_part)
            for layer in stripped_model.layers
            if layer.name in hls_config["LayerName"]
        }
        
        worst_rf = max(natural_rfs.values()) if natural_rfs else 0
        if worst_rf > target_rf:
            worst_layer = max(natural_rfs, key=natural_rfs.get)
            raise RuntimeError(
                f"Configured reuse_factor={target_rf} is below worst-layer "
                f"natural min RF: {worst_layer!r} requires {worst_rf} "
                f"(io_stream Conv needs RF >= kh*kw*n_chan). "
                f"Raise hls_config.reuse_factor in the config."
            )

        for layer in stripped_model.layers:
            if layer.name not in hls_config["LayerName"]:
                continue

            natural_rf = natural_rfs[layer.name]
            rf = target_rf if natural_rf == worst_rf else natural_rf
            hls_config["LayerName"][layer.name]["ReuseFactor"] = rf

            # Pin accumulator/result precision so hls4ml doesn't auto-widen.
            layer_cfg = hls_config["LayerName"][layer.name]
            if isinstance(layer_cfg.get("Precision"), dict):
                layer_cfg["Precision"]["accum"] = ctx.hls4ml.hls_config.precision
                layer_cfg["Precision"]["result"] = ctx.hls4ml.hls_config.precision

        hls_config["Model"]["Precision"] = ctx.hls4ml.hls_config.precision
        hls_config["Model"]["Strategy"] = ctx.hls4ml.hls_config.strategy

        hls_config["Model"]["ConvImplementation"] = "LineBuffer"

        # catch any per-layer precision narrower than
        # ap_fixed<32,16> on accum/result. No-op for current NauticML because
        # we pin precisions ourselves, but defensive for future overrides.
        HLSBuilder._widen_narrow_precisions(hls_config)

        # Guardrail: every compute layer must have a LayerName entry. Without
        # this, hls4ml silently uses Model defaults
        _MAC_CLASSES = ("Dense", "Conv1D", "Conv2D")
        missing = [
            l.name for l in stripped_model.layers
            if type(l).__name__ in _MAC_CLASSES and l.name not in hls_config["LayerName"]
        ]
        if missing:
            raise RuntimeError(
                f"hls_config missing LayerName entries for compute layers {missing}. "
                "This would cause silent fallback to Model defaults."
            )

        return hls_config, stripped_model

    @staticmethod
    def strip_for_hls(model):
        """
        Unwrap MCD, strip pruning wrappers, drop dropout layers. Asserts the
        topology is single-input / single-output / linear — Sequential rebuild
        can't represent anything else and we'd rather fail loud than produce
        a model with silently-dropped branches.
        """
        from tensorflow_model_optimization.sparsity.keras import strip_pruning
        from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer

        # 1. Extract the inner model if wrapped (e.g., MonteCarloDropoutModel).
        source = model
        if hasattr(source, "model") and isinstance(source.model, tf.keras.Model):
            source = source.model

        # # 2. Strip pruning wrappers if present. Tightened except: only swallow
        # # the specific "not a pruned model" case, not arbitrary errors.
        # has_pruning = any(
        #     isinstance(l, pruning_wrapper.PruneLowMagnitude) for l in source.layers
        # )
        # if has_pruning:
        source = strip_pruning(source)

        # 3. Topology guardrail: Sequential rebuild only works for linear models.
        if len(source.outputs) != 1:
            raise RuntimeError(
                f"strip_for_hls expects a single-output model, got {len(source.outputs)}"
            )
        
        # Topology guardrail: Sequential rebuild only works for single-output models.
        # Branching (multi-input layers like Add/Concatenate) will fail at clean.add()
        # with a clear Keras error, so we don't pre-check for it.
        if len(source.outputs) != 1:
            raise RuntimeError(
                f"strip_for_hls expects a single-output model, got {len(source.outputs)}"
            )
                            
        # 4. Rebuild as a clean Sequential backbone.
        clean = tf.keras.models.Sequential()
        input_shape = source.input_shape[1:]  # exclude batch dim

        for layer in source.layers:
            is_dropout = (
                isinstance(layer, (tf.keras.layers.Dropout, InferenceDropoutLayer))
                or "dropout" in layer.name.lower()
            )
            if is_dropout:
                continue

            config = layer.get_config()
            if len(clean.layers) == 0:
                config["batch_input_shape"] = (None,) + input_shape

            new_layer = layer.__class__.from_config(config)
            clean.add(new_layer)
            new_layer.set_weights(layer.get_weights())

        return clean

    @staticmethod
    def get_min_rf(layer, fpga_part):
        """Per-layer minimum reuse factor based on a rough DSP budget."""
        DSP_COUNT = {
            "xcku115-flvb2104-2-i": 5520,
            "xcu250-figd2104-2l-e": 12288,
            "xczu7ev-ffvc1156-2-e": 1728,
        }

        weights = layer.get_weights()
        if not weights:
            return 1

        total = np.prod(weights[0].shape)
        max_parallel = DSP_COUNT.get(fpga_part, 4096)

        min_rf = max(1, int(np.ceil(total / max_parallel)))
        rf = next(i for i in range(min_rf, total + 1) if total % i == 0)

        if isinstance(layer, tf.keras.layers.Conv2D):
            kh, kw = layer.kernel_size
            n_chan = layer.input_shape[-1]
            rf = max(rf, kh * kw * n_chan)
        elif isinstance(layer, tf.keras.layers.Conv1D):
            rf = max(rf, layer.kernel_size[0])

        return rf

    @staticmethod
    def _widen_narrow_precisions(hls_config):
        SAFE_PREC = "ap_fixed<24, 14>"
        WIDEN_KEYS = ("accum", "result")
        # Activations derive internal table-index types from accum/result;
        # widening these explodes table sizes past 32 bits.
        SKIP_NAMES = ("softmax", "sigmoid", "tanh", "relu")

        def _bits(prec_str):
            m = re.search(r"<\s*(\d+)\s*,\s*(\d+)", prec_str)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        for layer_name, layer_cfg in hls_config.get("LayerName", {}).items():
            if not isinstance(layer_cfg, dict):
                continue
            if any(s in layer_name.lower() for s in SKIP_NAMES):
                continue
            prec = layer_cfg.get("Precision")
            if not isinstance(prec, dict):
                continue
            for key in WIDEN_KEYS:
                cur = prec.get(key)
                if not isinstance(cur, str) or cur == "auto":
                    continue
                W, I = _bits(cur)
                if W < 32 or I < 16:
                    prec[key] = SAFE_PREC
        return hls_config

    @staticmethod
    def write_hls_model_bayes(ctx, hls_model):
        """
        Write the hls4ml project, stripping Bayesian artifacts from forker hls4ml
        and patching the mask_index ghost interface left by the Bayesian hls4ml fork.
        """
        hls_dir = ctx.hls4ml.hls_project_dir

        # Synthesise as deterministic — all Bayesian layers were stripped upstream.
        hls_model.config.config["Bayes"] = False
        hls_model.write()

        cpp_file = os.path.join(hls_dir, "firmware", "myproject.cpp")
        if os.path.exists(cpp_file):
            with open(cpp_file, "r") as f:
                lines = f.readlines()
            with open(cpp_file, "w") as f:
                for line in lines:
                    if "mask_index" in line:
                        f.write(f"// {line}")
                    else:
                        f.write(line)

        return hls_model

    # ------------------------------------------------------------------
    # Diagnostics + optional Vitis 2022.2+ TCL patch
    # ------------------------------------------------------------------

    @staticmethod
    def dump_hls_failure_log(log, project_dir, n_lines=120):
        """
        After a Vitis HLS failure the actual root cause is in vitis_hls.log
        inside the project — stdout only shows the generic error. Tails that
        file and csynth.log so the cause (clang OOM, partition errors, etc.)
        actually surfaces in the worker logs.
        """
        if not project_dir:
            return
        candidates = [
            os.path.join(project_dir, "myproject_prj", "solution1", "vitis_hls.log"),
            os.path.join(project_dir, "myproject_prj", "solution1", "syn", "report", "csynth.log"),
            os.path.join(project_dir, "vitis_hls.log"),
        ]
        found_any = False
        for path in candidates:
            if not os.path.exists(path):
                continue
            found_any = True
            try:
                with open(path, "r", errors="replace") as f:
                    lines = f.readlines()
            except OSError as e:
                log.warning(f"Could not read {path}: {e}")
                continue
            tail = lines[-n_lines:] if len(lines) > n_lines else lines
            log.error(f"=== Tail of {path} ({len(tail)} of {len(lines)} lines) ===")
            for line in tail:
                log.error(line.rstrip())
        if not found_any:
            log.warning(f"No Vitis HLS log files found under {project_dir}")

    @staticmethod
    def patch_build_prj(project_dir, log):
        
        """
        Patch hls4ml's generated build_prj.tcl for Vitis HLS 2022.2+ where
        `config_array_partition -maximum_size` was replaced by
        syn.array_partition.complete_threshold. Also strips cosim/export
        commands defensively. Off by default — gate via
        ctx.hls4ml.patch_vitis_2022_2.
        """
        candidates = list(project_dir.rglob("build_prj.tcl"))
        if not candidates:
            log.warning(f"no build_prj.tcl under {project_dir}; skipping patch")
            return

        patched_any = False
        for build_tcl in candidates:
            text = build_tcl.read_text()
            original = text

            text = re.sub(
                r"^\s*config_array_partition\s+-maximum_size\s+\d+\s*$",
                "# config_array_partition -maximum_size  (stripped: removed in Vitis HLS 2022.2+)",
                text, flags=re.MULTILINE,
            )
            text = re.sub(
                r"^\s*cosim_design\b.*$",
                "# cosim_design (stripped: cosim disabled)",
                text, flags=re.MULTILINE,
            )
            text = re.sub(
                r"^\s*export_design\b.*$",
                "# export_design (stripped: export disabled)",
                text, flags=re.MULTILINE,
            )
            text = re.sub(
                r"^\s*config_export\b.*$",
                "# config_export (stripped: export disabled)",
                text, flags=re.MULTILINE,
            )

            if text != original:
                build_tcl.write_text(text)
                log.info(f"Patched {build_tcl.relative_to(project_dir)}")
                patched_any = True

        if not patched_any:
            log.warning(
                f"No build_prj.tcl under {project_dir} contained "
                "maximum_size/cosim/export lines to strip"
            )