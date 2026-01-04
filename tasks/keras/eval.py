
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from sklearn.metrics import accuracy_score
from nautic import taskx

class KerasEval:

    @taskx
    def eval(ctx):
        ctx.eval.accuracy = KerasEval.evaluate_accuracy(ctx)
        ctx.eval.ece = KerasEval.evaluate_ece(ctx)
        ctx.eval.ape = KerasEval.evaluate_ape(ctx)
        ctx.eval.flops = KerasEval.evaluate_flops(ctx)

    def evaluate_ape(ctx) -> float:

        return 0.0

    def evaluate_ece(ctx) -> float:
        def entropy(output):
            batch_size = output.shape[0]
            entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
            return entropy

        x = ctx.dataset.data["x_test"]

        mean = x.mean(axis=(0, 1, 2), keepdims=True)
        std = x.std(axis=(0, 1, 2), keepdims=True)

        x_noise = np.random.normal(mean, std, size=x.shape).astype(x.dtype)

        return entropy(ctx.model.logic.predict(np.ascontiguousarray(x_noise)))

    def evaluate_accuracy(ctx):

        model = load_model(ctx.experiment.ckpt_file)
        y_prob = model.predict(ctx.dataset.data["x_test"])

        accuracy = float(accuracy_score(
            np.argmax(ctx.dataset.data["y_test"], axis=1),
            np.argmax(y_prob, axis=1)
        ))

        return accuracy


    def evaluate_flops(ctx):
        """
        Calculate FLOPS for tf.keras.Model or tf.keras.Sequential .
        Ignore operations used in only training mode such as Initialization.
        Use tf.profiler of tensorflow v1 api.
        """

        model = ctx.model.original

        batch_size = None
        if batch_size is None:
            batch_size = 1

        # convert tf.keras model into frozen graph to count FLOPS about operations used at inference
        # FLOPS depends on batch size
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
        ]

        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPS with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        return flops.total_float_ops