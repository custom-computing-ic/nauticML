"""Per-iteration GPU/CPU device selection for a shared machine.

The chosen GPU is contended, so each DSE iteration tries to acquire it at
training time (with retries) and records the device it settled on here.
Evaluation in the *same* iteration reads it back so train + eval agree on a
device, and the *next* iteration is free to try the GPU again.

We deliberately never hide the GPU globally: ``tf.config.set_visible_devices``
is irreversible once devices are initialised, which is exactly what made the
old approach drop to CPU permanently. Instead we only steer op placement with
``tf.device`` context managers, so a GPU that is busy this iteration can still
be picked up next iteration once it frees.
"""
import time

import tensorflow as tf
from tensorflow.keras import backend as K

GPU = "/GPU:0"
CPU = "/CPU:0"

# The family of TF errors a contended/held shared GPU raises: OOM
# (ResourceExhausted), generic device failures (Internal), Grappler/session
# creation failures (Unknown — "Failed to create session"), and uninitialised
# state (FailedPrecondition). Callers catch these to degrade to CPU uniformly.
GPU_ERRORS = (
    tf.errors.ResourceExhaustedError,
    tf.errors.InternalError,
    tf.errors.UnknownError,
    tf.errors.FailedPreconditionError,
)

# Retry policy for grabbing a busy GPU on the shared machine.
GPU_ACQUIRE_ATTEMPTS = 3
GPU_ACQUIRE_SLEEP_S = 10

# Device the current iteration settled on. Defaults to CPU so evaluation is
# safe even if it somehow runs before training has picked a device.
_current_device = CPU


def current_device():
    """Device string ('/GPU:0' or '/CPU:0') the current iteration is using."""
    return _current_device


def _gpu_usable():
    """True if a visible GPU can run a trivial op right now.

    A tiny matmul forces device init / Grappler cluster creation — the same
    paths that otherwise blow up deep inside training on a contended GPU — so
    this fails fast and cleanly while the GPU is busy or out of memory.
    """
    try:
        if not tf.config.get_visible_devices('GPU'):
            return False
        with tf.device(GPU):
            _ = tf.matmul(tf.ones((8, 8)), tf.ones((8, 8))).numpy()
        return True
    except Exception:
        return False


def acquire_device(log, attempts=GPU_ACQUIRE_ATTEMPTS, sleep_s=GPU_ACQUIRE_SLEEP_S):
    """Pick the device for this iteration, retrying a busy GPU before CPU.

    Probes the GPU up to ``attempts`` times (sleeping ``sleep_s`` seconds
    between tries). Returns GPU if a probe succeeds, otherwise CPU. The choice
    is recorded for :func:`current_device` so evaluation runs on the same
    device. Never hides the GPU, so the next iteration can try again.
    """
    global _current_device

    # No visible GPU at all -> straight to CPU, retrying would be pointless.
    if not tf.config.get_visible_devices('GPU'):
        _current_device = CPU
        return CPU

    for attempt in range(1, attempts + 1):
        if _gpu_usable():
            log.info(f"✅ GPU acquired for this iteration (attempt {attempt}/{attempts}).")
            _current_device = GPU
            return GPU
        log.warning(f"⚠️ GPU unavailable (attempt {attempt}/{attempts}).")
        if attempt < attempts:
            time.sleep(sleep_s)

    log.warning("Using CPU for this iteration (train + eval).")
    _current_device = CPU
    return CPU


def fallback_to_cpu(log=None):
    """Pin the rest of *this iteration* to CPU and release the GPU.

    Used when a GPU op fails partway through training or evaluation: the
    remaining work for this iteration runs on CPU, but the GPU is left visible
    so a later iteration can try it again.
    """
    global _current_device
    if _current_device != CPU:
        if log is not None:
            log.warning("Falling back to CPU for the rest of this iteration.")
        _current_device = CPU
        K.clear_session()
