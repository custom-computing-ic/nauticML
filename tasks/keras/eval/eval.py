from nautic import taskx

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
                state["model"] = load_model(ctx.experiment.ckpt_file, custom_objects=co)
            return state["model"]

        def get_y_prob():
            if state["y_prob"] is None:
                state["y_prob"] = get_model().predict(ctx.dataset.data["x_test"])
            return state["y_prob"]

        ctx.eval.accuracy = KerasEval._cached(ctx, cache, "accuracy",
            lambda: KerasEval.evaluate_accuracy(ctx, get_y_prob()))
        ctx.eval.ece = KerasEval._cached(ctx, cache, "ece",
            lambda: KerasEval.evaluate_ece(ctx, get_y_prob()))
        ctx.eval.ape = KerasEval._cached(ctx, cache, "ape",
            lambda: KerasEval.evaluate_ape(ctx, get_model))
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

    def evaluate_ape(ctx, get_model) -> float:
        def entropy(output):
            batch_size = output.shape[0]
            entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
            return entropy

        x = ctx.dataset.data["x_train"]

        # TODO: ask about mean - should be hard-coded?
        x_noise = np.random.normal(ctx.dataset.mean, ctx.dataset.std, size=x.shape).astype(x.dtype)

        output = get_model().predict(np.ascontiguousarray(x_noise))
        return entropy(output)

    def evaluate_accuracy(ctx, y_prob):
        accuracy = float(accuracy_score(
            np.argmax(ctx.dataset.data["y_test"], axis=1),
            np.argmax(y_prob, axis=1)
        ))

        return accuracy

    def evaluate_flops(ctx):
        """
        Calculate FLOPS for tf.keras.Model or tf.keras.Sequential.
        Ignore operations used in only training mode such as Initialization.

        Counted in-process via the TF v1 profiler. FLOPS is a soft metric, so
        any failure is swallowed and reported as 0.0 rather than crashing the DSE.
        """
        model = ctx.model.original
        try:
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