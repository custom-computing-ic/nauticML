import json
import os
from pathlib import Path

try:
    import fcntl  # POSIX-only; the cluster runs Linux. Degrade gracefully elsewhere.
except ImportError:  # pragma: no cover - Windows dev machines
    fcntl = None


def _unwrap(val):
    """Unwrap a nautic context value, which may be wrapped in a ref."""
    return val.get() if hasattr(val, "get") else val


# For every key field: how to pull it off the ctx, and how to format it so the
# resulting key string is stable across runs and platforms.
_FIELD_SPECS = {
    "model": (lambda ctx: _unwrap(ctx.model.name),                      str),
    "dr":    (lambda ctx: _unwrap(ctx.model.dropout_rate),              lambda v: round(float(v), 6)),
    "pr":    (lambda ctx: _unwrap(ctx.model.p_rate),                    lambda v: round(float(v), 6)),
    "nbl":   (lambda ctx: _unwrap(ctx.model.num_bayes_layer),           int),
    "sf":    (lambda ctx: _unwrap(ctx.model.scale_factor),              lambda v: round(float(v), 6)),
    "mc":    (lambda ctx: _unwrap(ctx.hls4ml.num_mc_samples),           int),
    "clk":   (lambda ctx: _unwrap(ctx.hls4ml.hls_config.clock_period),  lambda v: round(float(v), 6)),
    "fpga":  (lambda ctx: _unwrap(ctx.hls4ml.hls_config.fpga_part),     str),
    "rf":    (lambda ctx: _unwrap(ctx.hls4ml.hls_config.reuse_factor),  int),
    # Power proxy mode (ff | params | model_proxy). None in the default
    # Vivado-estimator mode, in which case it is omitted from the key so
    # estimator-mode entries stay backward-compatible (see make_key).
    "proxy": (lambda ctx: _unwrap(getattr(ctx.hls4ml, "proxy", None)),  str),
}

# Fields that are dropped from the key when their value is None, rather than
# making the whole key unbuildable. Keeps estimator-mode power keys identical
# to before the proxy knob existed.
_OPTIONAL_FIELDS = {"proxy"}

# Fields that make up the cache key for each metric. Software metrics
# (accuracy, ece, ...) only depend on the model and its tunable params, so
# their results are reused across FPGA targets. Power additionally depends on
# the target device / clock / reuse factor, so those are folded into its key.
BASE_FIELDS = ("model", "dr", "pr", "nbl", "sf", "mc")
POWER_FIELDS = BASE_FIELDS + ("clk", "fpga", "rf", "proxy")

METRIC_KEY_FIELDS = {
    "accuracy": BASE_FIELDS,
    "ece":      BASE_FIELDS,
    "ape":      BASE_FIELDS,
    "flops":    BASE_FIELDS,
    "power":    POWER_FIELDS,
}


class CacheManager:
    """A shared, file-backed cache for evaluation metrics.

    Every metric stores its results in the same JSON file, namespaced by metric
    name, and is keyed by the subset of the ctx that actually affects its value
    (see ``METRIC_KEY_FIELDS``). Probing/writing is concurrency-safe via an
    advisory file lock where available.

    Stored layout::

        {
          "accuracy": {"<base-key>": 0.987, ...},
          "power":    {"<power-key>": {"power": ..., "energy": ..., ...}, ...},
          ...
        }
    """

    def __init__(self, cache_path):
        self.cache_path = Path(cache_path)

    @staticmethod
    def make_key(metric, ctx):
        """Build the cache key string for ``metric`` from ``ctx``.

        Returns ``None`` if the required fields cannot be read off the ctx.
        """
        fields = METRIC_KEY_FIELDS.get(metric)
        if fields is None:
            raise KeyError(f"No cache key schema registered for metric {metric!r}")
        try:
            parts = []
            for name in fields:
                extract, fmt = _FIELD_SPECS[name]
                value = extract(ctx)
                if value is None and name in _OPTIONAL_FIELDS:
                    continue  # e.g. estimator mode: no proxy suffix
                parts.append(f"{name}={fmt(value)}")
        except Exception:
            return None
        return "|".join(parts)

    def _load(self, fileobj):
        raw = fileobj.read()
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None  # caller decides how to react to corruption

    def probe(self, metric, ctx):
        """Return the cached value for ``(metric, ctx)``, or ``None`` if absent."""
        key = self.make_key(metric, ctx)
        if key is None or not self.cache_path.exists():
            return None

        with open(self.cache_path, "r") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = self._load(f)
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        if not data:
            return None
        return data.get(metric, {}).get(key)

    def write(self, metric, ctx, value):
        """Store ``value`` for ``(metric, ctx)``. Returns ``True`` on success.

        The read-modify-write happens under an exclusive lock so concurrent
        workers don't clobber each other's entries.
        """
        key = self.make_key(metric, ctx)
        if key is None:
            return False

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.touch(exist_ok=True)

        with open(self.cache_path, "r+") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = self._load(f)
                if data is None:
                    data = {}  # corrupt file: reset rather than lose the new write

                data.setdefault(metric, {})[key] = value

                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return True


# Shared cache instance used by every metric. Lives alongside this module so
# all of tasks/keras/eval reads and writes the one file.
DEFAULT_CACHE_PATH = Path(__file__).parent / "metrics_cache.json"

_default_cache = None


def get_cache():
    """Return the process-wide shared CacheManager."""
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager(DEFAULT_CACHE_PATH)
    return _default_cache
