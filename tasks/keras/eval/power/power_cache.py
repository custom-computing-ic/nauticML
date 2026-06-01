from tasks.keras.eval.cache_manager import get_cache

# Power is just another metric in the shared CacheManager. This adapter keeps
# the existing PowerCache API and the power-specific value encoding (a missing
# power/energy is stored as -1 rather than null so a known failure is
# distinguishable from an absent entry once read back).
METRIC = "power"


class PowerCache:
    @staticmethod
    def probe_cache(ctx):
        entry = get_cache().probe(METRIC, ctx)
        if entry is None:
            return None

        power = entry.get("power")
        energy = entry.get("energy")

        return {
            "power":     None if power == -1 else power,
            "energy":    None if energy == -1 else energy,
            "resources": entry.get("resources") or {},
            "timing":    entry.get("timing") or {},
            "max_ii":    entry.get("max_ii"),
        }

    @staticmethod
    def write_cache(ctx, *, power, energy, resources, timing, max_ii):
        value = {
            "power":     -1 if power is None else power,
            "energy":    -1 if energy is None else energy,
            "resources": resources or {},
            "timing":    timing or {},
            "max_ii":    max_ii,
        }

        if not get_cache().write(METRIC, ctx, value):
            ctx.log.warning("Could not build cache key; skipping cache write")
