from pathlib import Path
import re

CONTROL_PORTS = {
    "ap_clk", "ap_rst", "ap_rst_n", "ap_start",
    "ap_done", "ap_ready", "ap_idle", "ap_continue", "ap_ce",
}

class PortParser:

    @staticmethod
    def parse(verilog_path):
        text = Path(verilog_path).read_text()

        if PortParser._is_io_stream(text):
            return PortParser._parse_io_stream(text)
        return PortParser._parse_io_parallel(text)

    @staticmethod
    def _is_io_stream(text):
        return bool(re.search(r"\b\w+_TDATA\b", text))

    @staticmethod
    def _detect_reset(text):
        """Returns (port_name, active_value, inactive_value)."""
        if re.search(r"^\s*input\s+ap_rst_n\s*;", text, re.M):
            return "ap_rst_n", "1'b0", "1'b1"
        return "ap_rst", "1'b1", "1'b0"

    @staticmethod
    def _detect_ap_continue(text):
        """True iff the DUT declares an `ap_continue` input port (ap_ctrl_chain)."""
        return bool(re.search(r"^\s*input\s+ap_continue\s*;", text, re.M))

    # ---------------- io_stream ----------------
    @staticmethod
    def _parse_io_stream(text):
        in_streams = []
        for m in re.finditer(r"^\s*input\s+\[(\d+):(\d+)\]\s+(\w+)_TDATA\s*;", text, re.M):
            high, low, name = int(m.group(1)), int(m.group(2)), m.group(3)
            in_streams.append({"name": name, "tdata_width": high - low + 1})

        out_streams = []
        for m in re.finditer(r"^\s*output\s+\[(\d+):(\d+)\]\s+(\w+)_TDATA\s*;", text, re.M):
            high, low, name = int(m.group(1)), int(m.group(2)), m.group(3)
            out_streams.append({"name": name, "tdata_width": high - low + 1})

        assert len(in_streams) == 1, f"Expected 1 input stream, found {len(in_streams)}"
        assert len(out_streams) == 1, f"Expected 1 output stream, found {len(out_streams)}"

        reset_port, reset_active, reset_inactive = PortParser._detect_reset(text)
        return {
            "io_type": "io_stream",
            "reset_port": reset_port,
            "reset_active": reset_active,
            "reset_inactive": reset_inactive,
            "has_ap_continue": PortParser._detect_ap_continue(text),
            "input_stream": in_streams[0],
            "output_stream": out_streams[0],
        }

    # ---------------- io_parallel ----------------
    @staticmethod
    def _parse_io_parallel(text):
        reset_port, reset_active, reset_inactive = PortParser._detect_reset(text)

        in_buses = []
        for m in re.finditer(r"^\s*input\s+\[(\d+):(\d+)\]\s+(\w+)\s*;", text, re.M):
            high, low, name = int(m.group(1)), int(m.group(2)), m.group(3)
            if name in CONTROL_PORTS:
                continue
            in_buses.append({"name": name, "width": high - low + 1})

        out_buses = []
        for m in re.finditer(r"^\s*output\s+\[(\d+):(\d+)\]\s+(\w+)\s*;", text, re.M):
            high, low, name = int(m.group(1)), int(m.group(2)), m.group(3)
            if name in CONTROL_PORTS or name.endswith("_ap_vld"):
                continue
            out_buses.append({"name": name, "width": high - low + 1})

        out_groups = PortParser._group_outputs(out_buses)

        assert len(in_buses) == 1, f"Expected 1 input bus, found {len(in_buses)}"
        assert len(out_groups) == 1, f"Expected 1 output group, found {len(out_groups)}"

        return {
            "io_type": "io_parallel",
            "reset_port": reset_port,
            "reset_active": reset_active,
            "reset_inactive": reset_inactive,
            "has_ap_continue": PortParser._detect_ap_continue(text),
            "input_port": in_buses[0],
            "output_ports": out_groups[0]["ports"],
            "output_width_each": out_groups[0]["ports"][0]["width"],
        }

    @staticmethod
    def _group_outputs(out_buses):
        groups = {}
        for port in out_buses:
            m = re.match(r"^(.*?)_(\d+)$", port["name"])
            key = m.group(1) if m else port["name"]
            groups.setdefault(key, []).append(port)

        result = []
        for prefix, ports in groups.items():
            def sort_key(p):
                m = re.search(r"_(\d+)$", p["name"])
                return int(m.group(1)) if m else 0
            ports.sort(key=sort_key)
            result.append({"prefix": prefix, "ports": ports})
        return result