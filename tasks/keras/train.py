import os
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from nautic import taskx

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tasks.keras.svhn.utils import CosineAnnealingScheduler
from tasks.keras import device as dev


def _fit_with_cpu_fallback(ctx, fit_fn):
    """Train this iteration on a GPU if one can be acquired, otherwise on CPU.

    On the shared machine the chosen GPU may be busy, so ``dev.acquire_device``
    retries it before giving up. The device it settles on is recorded so
    evaluation in the same iteration runs there too. If the GPU can't be
    acquired — or a GPU run dies with OOM — we fall back to CPU *for this
    iteration only*; the next iteration is free to try the GPU again.
    """
    log = ctx.log
    device = dev.acquire_device(log)

    if device == dev.GPU:
        try:
            with tf.device(dev.GPU):
                return fit_fn(ctx.model.logic)
        except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
            log.warning(
                f"GPU training failed ({type(e).__name__}: {e}). "
                "Falling back to CPU for this iteration."
            )
            dev.fallback_to_cpu(log)

    # CPU path: rebuild the model so its variables live on CPU, then fit there.
    # Local import to avoid a circular import at module load.
    from tasks.keras.model_factory import KerasModels
    with tf.device(dev.CPU):
        KerasModels.get_model(ctx)
        return fit_fn(ctx.model.logic)

class KerasTrain:
    @taskx
    def train_model(ctx):
        """
        Trains the model on the provided dataset using data augmentation, pruning, and custom learning rate schedulers.

        Args:
            args: Parsed command-line arguments.
            model: The Keras model to train.
            dataset: The dataset dictionary returned by get_dataset().
        """
        log = ctx.log
        dataset = ctx.dataset.data

        class ProgressCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch / nepoch) * 100
                log.artifact(progress_id=pg_id,
                                progress=progress)

        # Remove stale checkpoint from a previous DSE iteration whose
        # architecture (scale_factor, num_bayes_layer) may differ.
        ckpt = ctx.experiment.ckpt_file
        if os.path.exists(ckpt):
            if os.path.isdir(ckpt):
                shutil.rmtree(ckpt)
            else:
                os.remove(ckpt)

        if ctx.model.name == "lenet":
            chkp = ModelCheckpoint(
                ctx.experiment.ckpt_file,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
            )

            if ctx.model.p_rate != 0.0:
                callbacks = [chkp, pruning_callbacks.UpdatePruningStep() ]

            else:
                callbacks = [chkp]

            pg_id = log.artifact(
                progress=0.0,
                description="Performing training for LeNet",
            )

            nepoch = ctx.train.num_epoch - 1
            callbacks.append(ProgressCallback())

            def _fit_lenet(m):
                return m.fit(
                    dataset['x_train'],
                    dataset['y_train'],
                    batch_size=ctx.train.batch_size,
                    epochs=ctx.train.num_epoch,
                    initial_epoch=1,
                    validation_split=ctx.train.validation_split,
                    callbacks=callbacks)

            train_stat = _fit_with_cpu_fallback(ctx, _fit_lenet)

            history = {k: [float(v) for v in vals] for k, vals in train_stat.history.items()}
            log.artifact(table=history,
                         key=f"train-results-{ctx.train.id.get()}",
                         description="Training results")

        elif ctx.model.name == "resnet":
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                # rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=4,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=4,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
            )

            train_gen = datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=ctx.train.batch_size)
            reduce_lr = CosineAnnealingScheduler(T_max=ctx.train.num_epoch, eta_max=ctx.train.learning_rate, eta_min=1e-4)

            chkp = ModelCheckpoint(
                ctx.experiment.ckpt_file,                
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
                save_freq="epoch",
            )

            if ctx.model.p_rate != 0.0:
                callbacks = [reduce_lr, chkp,  pruning_callbacks.UpdatePruningStep() ]
            else:
                callbacks = [reduce_lr, chkp]

            pg_id = log.artifact(
                progress=0.0,
                description="Performing training for ResNet",
            )

            nepoch = ctx.train.num_epoch - 1
            callbacks.append(ProgressCallback())

            def _fit_resnet(m):
                return m.fit(train_gen,
                    epochs=ctx.train.num_epoch,
                    callbacks=callbacks,
                    validation_data=(dataset['x_val'], dataset['y_val']),
                    )

            train_stat = _fit_with_cpu_fallback(ctx, _fit_resnet)

            history = {k: [float(v) for v in vals] for k, vals in train_stat.history.items()}
            log.artifact(table=history,
                         key=f"train-results-{ctx.train.id.get()}",
                         description="Training results")