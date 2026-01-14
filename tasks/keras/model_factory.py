
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.optimizers import SGD
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.models import clone_model


from tasks.keras.models.lenet import LeNet
from nautic import taskx
from tasks.keras.models.resnet import ResNet18

class KerasModels:
    @taskx
    def get_model(ctx):
        """
        Constructs the model architecture based on the selected model and configuration.

        Args:
            args: Parsed command-line arguments.

        Returns:
            model: A compiled Keras model.
        """

        factory_quant = {   }

        factory_nquant = {
            "lenet": LeNet,
            "resnet": ResNet18
        }

        if ctx.model.is_quant:
            model_builder = factory_quant.get(ctx.model.name, None)
        else:
            model_builder = factory_nquant.get(ctx.model.name, None)

        if not model_builder:
            raise NotImplementedError(f"Model '{ctx.model.name}' (quant: {ctx.model.is_quant}) not supported")

        ctx.model.logic = model_builder(ctx)
        ctx.model.original = clone_model(ctx.model.logic)

        KerasModels.prune(ctx)

    @staticmethod
    def prune(ctx):
        if ctx.model.p_rate == 0.0:
            return

        model = ctx.model.logic
        x_train_len = len(ctx.dataset.data['x_train'])
        NSTEPS =   int(x_train_len) // ctx.train.batch_size

        def pruneFunction(layer):
            pruning_params = {
                'pruning_schedule': sparsity.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=ctx.model.p_rate,
                    begin_step=NSTEPS * 2,
                    end_step=NSTEPS * 8,
                    frequency=NSTEPS
                )
            }
            if isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

            if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'fc_2': # exclude output_dense
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer

        #print_qmodel_summary(model)
        model = tf.keras.models.clone_model(model, clone_function=pruneFunction)

        model.compile(optimizer=SGD(learning_rate = ctx.train.learning_rate),
                    loss=['categorical_crossentropy'], metrics=['accuracy'])

        ctx.model.logic = model

