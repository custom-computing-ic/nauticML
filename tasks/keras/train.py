from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from nautic import taskx

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
        model = ctx.model.logic
        dataset = ctx.dataset.data

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
                description="Performing training",
            )

            nepoch = ctx.train.num_epoch - 1
            class ProgressCallback(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch / nepoch) * 100
                    log.artifact(progress_id=pg_id,
                                 progress=progress)


            callbacks.append(ProgressCallback())

            train_stat = model.fit(
                dataset['x_train'],
                dataset['y_train'],
                batch_size=ctx.train.batch_size,
                epochs=ctx.train.num_epoch,
                initial_epoch=1,
                validation_split=ctx.train.validation_split,
                callbacks=callbacks)

            log.artifact(table=train_stat.history,
                         key=f"train-results-{ctx.train.id.get()}",
                         description="Training results",)




