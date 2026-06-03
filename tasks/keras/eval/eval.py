from nautic import taskx

import os
import re
import sys
import shutil
import tempfile
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from sklearn.metrics import accuracy_score
from logic.converter.keras.dropout.inference_layer import BayesianDropout
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from tasks.keras.trust.converter.dropout.mc_model import MonteCarloDropoutModel

from tasks.keras.eval.power.power_eval import KerasEnergy
from tasks.keras.eval.power.power_cache import PowerCache
from tasks.keras.eval.cache_manager import get_cache
from tasks.keras import device as dev

class KerasEval:

    @taskx
    def eval(ctx):

        co = {  "BayesianDropout": BayesianDropout,
            "MonteCarloDropoutModel": MonteCarloDropoutModel,
            "PruneLowMagnitude": pruning_wrapper.PruneLowMagnitude
        }

        cache = get_cache()

        # Loading the checkpoint and running predict are the expensive steps, so
        # defer them until a cache miss actually needs them. A run where every
        # metric is cached touches neither.
        state = {"model": None, "y_prob": None}

        def get_model():
            if state["model"] is None:
                with tf.device(dev.current_device()):
                    state["model"] = load_model(ctx.experiment.ckpt_file, custom_objects=co)
            return state["model"]

        def on_device_predict(predict_call):
            """Run predict_call(model) on this iteration's device.

            If the shared GPU dies mid-evaluation, pin the rest of the
            iteration to CPU, reload the model there and retry — so a contended
            GPU never fails the eval stage.
            """
            try:
                with tf.device(dev.current_device()):
                    return predict_call(get_model())
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                if dev.current_device() == dev.CPU:
                    raise
                ctx.log.warning(
                    f"GPU eval failed ({type(e).__name__}: {e}). Retrying on CPU."
                )
                dev.fallback_to_cpu(ctx.log)
                state["model"] = None  # force a fresh load on CPU
                with tf.device(dev.CPU):
                    return predict_call(get_model())

        def get_y_prob():
            if state["y_prob"] is None:
                state["y_prob"] = on_device_predict(
                    lambda m: m.predict(ctx.dataset.data["x_test"]))
            return state["y_prob"]

        ctx.eval.accuracy = KerasEval._cached(ctx, cache, "accuracy",
            lambda: KerasEval.evaluate_accuracy(ctx, get_y_prob()))
        ctx.eval.ece = KerasEval._cached(ctx, cache, "ece",
            lambda: KerasEval.evaluate_ece(ctx, get_y_prob()))
        ctx.eval.ape = KerasEval._cached(ctx, cache, "ape",
            lambda: KerasEval.evaluate_ape(ctx, on_device_predict))
        ctx.eval.flops = KerasEval._cached(ctx, cache, "flops",
            lambda: KerasEval.evaluate_flops(ctx))

        KerasEnergy.evaluate_energy(ctx, get_model)
        # TODO: decouple these evaluations with a map of things to update and the acc function, and do the same in bayes opt for logging
        # TODO: decouple also the pareto frontier
        # TODO: add also latency to the pareto front generated

    @staticmethod
    def _cached(ctx, cache, metric, compute):
        """Return the cached metric value, or compute it on a miss and store it."""
        hit = cache.probe(metric, ctx)
        if hit is not None:
            ctx.log.info(f"Using cached {metric}={hit}")
            return hit

        value = compute()
        try:
            cache.write(metric, ctx, value)
        except Exception as e:
            ctx.log.warning(f"Failed to write {metric} cache: {e}")
        return value

    @taskx
    def probe_cache(ctx):
        """Populate ctx.eval.* straight from cache for the current BO point.

        Runs right after a Bayesian point is chosen. If *every* metric for this
        configuration is already cached, the metrics are written onto ctx and
        ctx.eval.cached is set True so the caller can skip training + evaluation
        and ask the optimiser for the next point. Otherwise ctx.eval.cached is
        set False and nothing else is touched.
        """
        cache = get_cache()

        accuracy = cache.probe("accuracy", ctx)
        ece = cache.probe("ece", ctx)
        ape = cache.probe("ape", ctx)
        flops = cache.probe("flops", ctx)
        # Power probe returns both power and energy (or None on a miss). A
        # previously-failed eval is still a hit (stored as a sentinel) so we
        # don't pointlessly retrain a configuration that can't be built.
        power_entry = PowerCache.probe_cache(ctx)

        if None in (accuracy, ece, ape, flops) or power_entry is None:
            ctx.eval.cached = False
            return

        ctx.eval.accuracy = accuracy
        ctx.eval.ece = ece
        ctx.eval.ape = ape
        ctx.eval.flops = flops
        ctx.eval.power = power_entry["power"]
        ctx.eval.energy = power_entry["energy"]
        ctx.eval.cached = True

        ctx.log.info(
            "Full cache hit for this configuration "
            f"(accuracy={accuracy}, ece={ece}, ape={ape}, flops={flops}, "
            f"power={power_entry['power']}, energy={power_entry['energy']}); "
            "skipping train + eval"
        )

    def evaluate_ece(ctx, y_prob) -> float:
        y_logits    = np.log(y_prob/(1-y_prob + 1e-15))

        # we use CPU device as TF_DETERMINISTIC_OPS is not implemented in tf.math.bincount(x)
        with tf.device('/CPU:0'):
            ece_keras   = tfp.stats.expected_calibration_error(num_bins=ctx.eval.num_bins,
                logits=y_logits, labels_true=np.argmax(ctx.dataset.data["y_test"],axis=1), labels_predicted=np.argmax(y_prob,axis=1))
            
        return float(ece_keras)

    def evaluate_ape(ctx, on_device_predict) -> float:
        def entropy(output):
            batch_size = output.shape[0]
            entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
            return entropy

        x = ctx.dataset.data["x_train"]

        # TODO: ask about mean - should be hard-coded?
        x_noise = np.random.normal(ctx.dataset.mean, ctx.dataset.std, size=x.shape).astype(x.dtype)

        output = on_device_predict(lambda m: m.predict(np.ascontiguousarray(x_noise)))
        return entropy(output)

    def evaluate_accuracy(ctx, y_prob):
        accuracy = float(accuracy_score(
            np.argmax(ctx.dataset.data["y_test"], axis=1),
            np.argmax(y_prob, axis=1)
        ))

        return accuracy

    def evaluate_flops(ctx):
        """
        Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
        Ignore operations used in only training mode such as Initialization.

        The profiler path uses convert_variables_to_constants_v2_as_graph, which
        builds a Grappler cluster that probes the GPU even under
        tf.device('/CPU:0') and dies with UnknownError("Failed to create
        session") on a contended shared GPU. To keep it off the GPU entirely we
        run the count in a short-lived subprocess with CUDA_VISIBLE_DEVICES=""
        (set before TF imports), which forces a CPU-only Grappler cluster and
        yields identical numbers. See flops_worker.py.
        """
        model = ctx.model.original

        tmpdir = tempfile.mkdtemp(prefix="flops_")
        model_path = os.path.join(tmpdir, "model.h5")
        worker = os.path.join(os.path.dirname(__file__), "flops_worker.py")
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
        )
        try:
            model.save(model_path)

            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""  # GPU invisible to the subprocess
            env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

            proc = subprocess.run(
                [sys.executable, worker, model_path],
                cwd=repo_root, env=env,
                capture_output=True, text=True, timeout=600,
            )
            match = re.search(r"FLOPS_RESULT=(\d+)", proc.stdout)
            if match:
                return int(match.group(1))

            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
            ctx.log.warning(
                f"FLOPS subprocess returned no result (exit {proc.returncode}). "
                f"Output tail:\n{tail}"
            )
        except Exception as e:
            ctx.log.warning(
                f"FLOPS subprocess failed ({type(e).__name__}: {e}). "
                "Trying in-process on CPU."
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Last resort: in-process count on CPU. This may still hit the
        # Grappler/GPU issue on a contended machine, so swallow it and return
        # 0.0 rather than crash the whole DSE — FLOPS is a soft metric.
        try:
            with tf.device(dev.CPU):
                inputs = [
                    tf.TensorSpec([1] + inp.shape[1:], inp.dtype)
                    for inp in model.inputs
                ]
                real_model = tf.function(model).get_concrete_function(inputs)
                frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)
                run_meta = tf.compat.v1.RunMetadata()
                opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.compat.v1.profiler.profile(
                    graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
                )
            return flops.total_float_ops
        except Exception as e:
            ctx.log.error(
                f"FLOPS count unavailable ({type(e).__name__}: {e}); returning 0.0"
            )
            return 0.0