"""Standalone FLOPS counter, run as a subprocess with the GPU hidden.

The in-process FLOPS path calls ``convert_variables_to_constants_v2_as_graph``,
which builds a Grappler optimisation cluster (``gcluster.Cluster()`` ->
``TF_NewCluster``) that *probes the GPU regardless of any tf.device('/CPU:0')
context*. On a contended shared GPU that probe dies with
``UnknownError: Failed to create session``.

Running the same computation here, in a fresh process with
``CUDA_VISIBLE_DEVICES=""`` exported *before* TensorFlow is imported, forces a
CPU-only Grappler cluster. It produces identical FLOPS numbers and never
touches the shared GPU.

Usage:
    python flops_worker.py <model_path>
Prints a single line ``FLOPS_RESULT=<int>`` on success.
"""
import os

# Must be set before TensorFlow is imported so CUDA is never initialised here.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)


def _custom_objects():
    """Custom layers that may appear in a saved NauticML model."""
    co = {}
    try:
        from tasks.keras.trust.converter.dropout.inference_layer import BayesianDropout
        co["BayesianDropout"] = BayesianDropout
    except Exception:
        pass
    try:
        from tensorflow_model_optimization.python.core.sparsity.keras import (
            pruning_wrapper,
        )
        co["PruneLowMagnitude"] = pruning_wrapper.PruneLowMagnitude
    except Exception:
        pass
    return co


def count_flops(model_path):
    model = tf.keras.models.load_model(
        model_path, custom_objects=_custom_objects(), compile=False
    )

    batch_size = 1
    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )
    return int(flops.total_float_ops)


if __name__ == "__main__":
    result = count_flops(sys.argv[1])
    # Unique marker so the parent can parse this out of the profiler's own
    # verbose stdout report.
    print(f"FLOPS_RESULT={result}")
