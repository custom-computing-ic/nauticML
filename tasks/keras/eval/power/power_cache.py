import json
import os
import fcntl
from pathlib import Path

POWER_CACHE_PATH = Path(__file__).parent / "power_cache.json"

class PowerCache:
    @staticmethod
    def _get(val):
        """Unwrap nautic context values which may be wrapped."""
        return val.get() if hasattr(val, "get") else val
    
    @staticmethod
    def _make_cache_key(ctx):
        try:
            model_name = PowerCache._get(ctx.model.name)

            # BO tunables
            dropout_rate    = PowerCache._get(ctx.model.dropout_rate)
            p_rate          = PowerCache._get(ctx.model.p_rate)
            num_bayes_layer = PowerCache._get(ctx.model.num_bayes_layer)
            scale_factor    = PowerCache._get(ctx.model.scale_factor)

            # HLS / hardware config
            mc_samples   = PowerCache._get(ctx.hls4ml.num_mc_samples)
            clock_period = PowerCache._get(ctx.hls4ml.hls_config.clock_period)
            fpga_part    = PowerCache._get(ctx.hls4ml.hls_config.fpga_part)
            reuse_factor = PowerCache._get(ctx.hls4ml.hls_config.reuse_factor)
        except Exception:
            return None

        return (
            f"model={model_name}"
            f"|dr={round(float(dropout_rate), 6)}"
            f"|pr={round(float(p_rate), 6)}"
            f"|nbl={int(num_bayes_layer)}"
            f"|sf={round(float(scale_factor), 6)}"
            f"|mc={int(mc_samples)}"
            f"|clk={round(float(clock_period), 6)}"
            f"|fpga={fpga_part}"
            f"|rf={int(reuse_factor)}"
        )
    
    @staticmethod
    def probe_cache(ctx):
        key = PowerCache._make_cache_key(ctx)
        if key is None:
            return None

        if not POWER_CACHE_PATH.exists():
            return None

        with open(POWER_CACHE_PATH, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    return None
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        entry = data.get(key)
        
        if entry is None:
            return None

        power  = entry.get("power")
        energy = entry.get("energy")

        return {
            "power":     None if power  == -1 else power,
            "energy":    None if energy == -1 else energy,
            "resources": entry.get("resources") or {},
            "timing":    entry.get("timing") or {},
            "max_ii":    entry.get("max_ii"),
        }

    @staticmethod
    def write_cache(ctx, *, power, energy, resources, timing, max_ii):
        key = PowerCache._make_cache_key(ctx)
        if key is None:
            ctx.log.warning("Could not build cache key; skipping cache write")
            return

        POWER_CACHE_PATH.touch(exist_ok=True)

        with open(POWER_CACHE_PATH, "r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                raw = f.read()
                try:
                    data = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    ctx.log.warning(
                        f"power_cache.json corrupt; resetting. content was: {raw[:200]}"
                    )
                    data = {}

                data[key] = {
                    "power":     -1 if power  is None else power,
                    "energy":    -1 if energy is None else energy,
                    "resources": resources or {},
                    "timing":    timing or {},
                    "max_ii":    max_ii,
                }

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)