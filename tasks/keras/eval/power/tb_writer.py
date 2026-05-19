from pathlib import Path

from tasks.keras.eval.power.port_parser import PortParser
import os

import re

import numpy as np

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Floors used when no design-derived estimate is supplied. The dynamic
# estimator below should usually overrule these.
DEFAULT_TIMEOUT_IO_PARALLEL = 200_000
DEFAULT_TIMEOUT_IO_STREAM   = 10_000_000

# Default SAIF sample count. Matches the value used in the sweep code; tune
# via ctx.hls4ml.power_mode.num_samples if vectorless mode is on.
N_SAIF_SAMPLES_DEFAULT = 64

# Default TB clock period in ns. Falls back to ctx.hls4ml.hls_config.clock_period
# when present; this constant exists only so the TB writer never sees None.
CLK_PERIOD_DEFAULT = 5.0

class TestbenchWriter:

    @staticmethod
    def get_clock_period(ctx):
        return ctx.hls4ml.hls_config.clock_period or CLK_PERIOD_DEFAULT

    @staticmethod
    def _reset_polarity(reset_port):
        """Active-low if name ends with _n, else active-high."""
        if reset_port.endswith("_n"):
            return "1'b0", "1'b1"
        return "1'b1", "1'b0"

    @staticmethod
    def _ap_continue_binding(ports):
        """
        Verilog text to splice into the DUT instantiation. Only emitted
        when the DUT actually exposes `ap_continue` (ap_ctrl_chain); for
        ap_ctrl_hs the port doesn't exist and binding it would fail
        elaboration. Tying it high acknowledges every ap_done immediately
        so back-to-back calls aren't blocked.
        """
        if ports.get("has_ap_continue"):
            return "\n        .ap_continue (1'b1),"
        return ""

    @staticmethod
    def estimate_timeout_cycles(qmodel, hls_config, n_samples, io_type,
                                safety=2, floor=200_000):
        """
        Conservative cycle budget for the power TB based on max layer II.

        Streaming Resource strategy II model:
          Dense: II_per_sample ≈ ReuseFactor (one inference per RF cycles)
          Conv:  II_per_sample ≈ output_positions × ReuseFactor
                 (Resource serialises MACs within an output position; output
                  positions are processed sequentially when
                  ParallelizationFactor is 1, which it is in our flow.)

        Total TB budget = max_layer_II × n_samples × safety + pipeline fill.
        """
        DENSE  = ("Dense",  "QDense")
        CONV1D = ("Conv1D", "QConv1D")
        CONV2D = ("Conv2D", "QConv2D")

        layer_cfgs = hls_config.get("LayerName", {}) or {}
        model_rf   = hls_config.get("Model", {}).get("ReuseFactor", 1) or 1

        def _flat_shape(s):
            if isinstance(s, list):
                s = s[0]
            return tuple(d for d in s if d is not None)

        def _output_positions(out_shape):
            if not out_shape:
                return 1
            spatial = out_shape[:-1] if len(out_shape) > 1 else ()
            prod = 1
            for d in spatial:
                prod *= max(1, int(d))
            return prod

        max_ii = 1
        for layer in qmodel.layers[1:]:
            cls = type(layer).__name__
            if cls not in DENSE + CONV1D + CONV2D:
                continue
            rf_cfg = layer_cfgs.get(layer.name, {})
            rf = rf_cfg.get("ReuseFactor", model_rf) if isinstance(rf_cfg, dict) else model_rf
            try:
                rf = max(1, int(rf))
            except (TypeError, ValueError):
                rf = max(1, int(model_rf))

            if cls in DENSE:
                ii = max(1, rf)
            else:
                ii = _output_positions(_flat_shape(layer.output_shape)) * rf
            if ii > max_ii:
                max_ii = ii

        fill = 50_000 if io_type == "io_stream" else 10_000
        return max(floor, max_ii * int(n_samples) * int(safety) + fill)

    @staticmethod
    def write_io_parallel(tb_path, *, dut_name, clk_period, in_width_total,
                          out_width_total, n_samples, ports,
                          timeout_cycles=DEFAULT_TIMEOUT_IO_PARALLEL):
        template = (TEMPLATE_DIR / "power_tb_io_parallel.v.tpl").read_text()

        out_decls = "\n".join(
            f"    wire [{p['width']-1}:0] {p['name']}_w;\n"
            f"    wire {p['name']}_ap_vld_w;"
            for p in ports["output_ports"]
        )

        out_conns = "".join(
            f",\n        .{p['name']}({p['name']}_w)"
            for p in ports["output_ports"]
        ) + "".join(
            f",\n        .{p['name']}_ap_vld({p['name']}_ap_vld_w)"
            for p in ports["output_ports"]
        )

        out_concat = ", ".join(
            f"{p['name']}_w" for p in reversed(ports["output_ports"])
        )

        capture_vld_expr = f"{ports['output_ports'][0]['name']}_ap_vld_w"

        reset_port = ports["reset_port"]
        reset_active = ports.get("reset_active")
        reset_inactive = ports.get("reset_inactive")
        if reset_active is None or reset_inactive is None:
            reset_active, reset_inactive = TestbenchWriter._reset_polarity(reset_port)

        body = template.format(
            dut_name=dut_name,
            clk_period=clk_period,
            in_width_total=in_width_total,
            out_width_total=out_width_total,
            n_samples=n_samples,
            output_wire_decls=out_decls,
            input_port_name=ports["input_port"]["name"],
            output_port_connections=out_conns,
            output_concat=out_concat,
            capture_vld_expr=capture_vld_expr,
            timeout_cycles=timeout_cycles,
            reset_port=reset_port,
            reset_active=reset_active,
            reset_inactive=reset_inactive,
            ap_continue_binding=TestbenchWriter._ap_continue_binding(ports),
        )
        Path(tb_path).write_text(body)
        return tb_path

    @staticmethod
    def write_io_stream(tb_path, *, dut_name, clk_period, n_samples,
                        beats_per_sample, ports,
                        timeout_cycles=DEFAULT_TIMEOUT_IO_STREAM):
        template = (TEMPLATE_DIR / "power_tb_io_stream.v.tpl").read_text()
        reset_port = ports["reset_port"]
        reset_active, reset_inactive = TestbenchWriter._reset_polarity(reset_port)

        body = template.format(
            dut_name=dut_name,
            clk_period=clk_period,
            in_tdata_width=ports["input_stream"]["tdata_width"],
            out_tdata_width=ports["output_stream"]["tdata_width"],
            n_samples=n_samples,
            beats_per_sample=beats_per_sample,
            in_port=ports["input_stream"]["name"],
            out_port=ports["output_stream"]["name"],
            reset_port=reset_port,
            reset_active=reset_active,
            reset_inactive=reset_inactive,
            ap_continue_binding=TestbenchWriter._ap_continue_binding(ports),
            timeout_cycles=timeout_cycles,
        )
        Path(tb_path).write_text(body)
        return tb_path
    
    # ------------------------------------------------------------------
    # SAIF testbench data generation
    # ------------------------------------------------------------------

    @staticmethod
    def _n_saif_samples(ctx):
        return getattr(ctx.hls4ml.power_mode, "num_samples", N_SAIF_SAMPLES_DEFAULT)

    @staticmethod
    def load_saif_testcases(ctx, stripped_model, project_dir, hls_config, io_type):
        """
        Generate both .npy reference vectors (for hls4ml's csim/cosim) and a
        hex-packed input_vectors.dat (for the custom Verilog power TB).
        Uses ctx.dataset.data["x_test"] — NauticML always has real data, so
        we never fall back to the Gaussian-mixture synthetic pipeline from
        the sweep.
        """
        os.makedirs(project_dir, exist_ok=True)

        n_samples = TestbenchWriter._n_saif_samples(ctx)
        x_sample = ctx.dataset.data["x_test"][:n_samples]
        x_sample = np.ascontiguousarray(x_sample.astype(np.float32))
        y_sample = stripped_model.predict(x_sample)

        input_tb_path = os.path.join(project_dir, "input_features.npy")
        output_tb_path = os.path.join(project_dir, "output_predictions.npy")
        np.save(input_tb_path, x_sample)
        np.save(output_tb_path, y_sample)

        prec = TestbenchWriter._input_precision(hls_config, stripped_model)
        W, I = TestbenchWriter._parse_fixed(prec)

        hex_input_path = os.path.join(project_dir, "input_vectors.dat")
        if io_type == "io_stream":
            assert x_sample.ndim in (3, 4), (
                f"io_stream expects 3D (Conv1D) or 4D (Conv2D) input, got {x_sample.shape}"
            )
            TestbenchWriter._to_hex_vectors_stream(x_sample, W, I, hex_input_path)
            beats_per_sample = int(np.prod(x_sample.shape[1:-1]))
        else:
            x_flat = x_sample.reshape(x_sample.shape[0], -1)
            TestbenchWriter._to_hex_vectors(x_flat, W, I, hex_input_path)
            beats_per_sample = 1

        return input_tb_path, output_tb_path, hex_input_path, W, beats_per_sample

    @staticmethod
    def _parse_fixed(precision_str):
        m = re.search(r"<\s*(\d+)\s*,\s*(\d+)", precision_str)
        if not m:
            raise ValueError(f"Cannot parse precision: {precision_str!r}")
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _input_precision(hls_config, model):
        default = hls_config["Model"]["Precision"]
        if isinstance(default, dict):
            default = default.get("default", "fixed<16,6>")

        input_name = model.layers[0].name
        per_layer = (
            hls_config.get("LayerName", {})
            .get(input_name, {})
            .get("Precision", {})
        )
        prec = per_layer.get("result") if isinstance(per_layer, dict) else per_layer

        if not prec or prec == "auto" or not isinstance(prec, str):
            return default
        if not (
            prec.startswith("fixed<") or prec.startswith("ap_fixed<")
            or prec.startswith("ufixed<") or prec.startswith("ap_ufixed<")
        ):
            return default
        return prec

    @staticmethod
    def _to_hex_vectors(samples, total_width, int_width, out_path):
        """Flat packing for io_parallel: feature i at bits [W*(i+1)-1 : W*i]."""
        frac_width = total_width - int_width
        scale = 1 << frac_width
        max_val = (1 << (total_width - 1)) - 1
        min_val = -(1 << (total_width - 1))
        mask = (1 << total_width) - 1
        n_features = samples.shape[1]
        hex_digits = (total_width * n_features + 3) // 4

        with open(out_path, "w") as f:
            for sample in samples:
                packed = 0
                for i, v in enumerate(sample):
                    fixed = int(np.round(float(v) * scale))
                    fixed = max(min(fixed, max_val), min_val)
                    packed |= (fixed & mask) << (i * total_width)
                f.write(f"{packed:0{hex_digits}x}\n")

    @staticmethod
    def _to_hex_vectors_stream(samples, total_width, int_width, out_path):
        """
        Per-beat packing for io_stream. Emits N*prod(spatial_dims) lines,
        each line containing C channels packed into one hex value.
        """
        frac_width = total_width - int_width
        scale = 1 << frac_width
        max_val = (1 << (total_width - 1)) - 1
        min_val = -(1 << (total_width - 1))
        mask = (1 << total_width) - 1

        n_channels = samples.shape[-1]
        hex_digits = (total_width * n_channels + 3) // 4

        n_samples = samples.shape[0]
        beats_per_sample = int(np.prod(samples.shape[1:-1]))
        samples_flat = samples.reshape(n_samples, beats_per_sample, n_channels)

        with open(out_path, "w") as f:
            for sample in samples_flat:
                for beat in sample:
                    packed = 0
                    for i, v in enumerate(beat):
                        fixed = int(np.round(float(v) * scale))
                        fixed = max(min(fixed, max_val), min_val)
                        packed |= (fixed & mask) << (i * total_width)
                    f.write(f"{packed:0{hex_digits}x}\n")

    # ------------------------------------------------------------------
    # Verilog testbench writing
    # ------------------------------------------------------------------

    @staticmethod
    def write_power_testbench(
        ctx, project_dir, stripped_model, hls_config, io_type,
        beats_per_sample, n_samples,
    ):
        """
        Parse the synthesised Verilog port map, then emit power_tb.v alongside
        the project. The downstream Vivado SAIF flow drives this TB.
        """
        verilog_dir = project_dir / "myproject_prj" / "solution1" / "syn" / "verilog"
        myproject_v = verilog_dir / "myproject.v"
        if not myproject_v.exists():
            raise FileNotFoundError(f"Expected synthesised Verilog at {myproject_v}")

        ports = PortParser.parse(myproject_v)
        tb_path = project_dir / "power_tb.v"

        timeout_cycles = TestbenchWriter.estimate_timeout_cycles(
            stripped_model, hls_config, n_samples, io_type,
        )
        ctx.log.info(f"TB timeout_cycles={timeout_cycles}")

        clk_period = TestbenchWriter.get_clock_period(ctx)

        if io_type == "io_parallel":
            out_width_each = ports["output_width_each"]
            n_outputs = len(ports["output_ports"])
            TestbenchWriter.write_io_parallel(
                tb_path,
                dut_name="myproject",
                clk_period=clk_period,
                in_width_total=ports["input_port"]["width"],
                out_width_total=out_width_each * n_outputs,
                n_samples=n_samples,
                ports=ports,
                timeout_cycles=timeout_cycles,
            )
        else:
            TestbenchWriter.write_io_stream(
                tb_path,
                dut_name="myproject",
                clk_period=clk_period,
                n_samples=n_samples,
                beats_per_sample=beats_per_sample,
                ports=ports,
                timeout_cycles=timeout_cycles,
            )

        ctx.log.info(f"Wrote {io_type} TB to {tb_path}")
        return tb_path
