import os
import tensorflow as tf
import random
import numpy as np
from nautic import taskx

class KerasExperiment:

    @taskx
    def initialize_experiment(ctx):
        def configure_gpus(gpu_indices):
            """
            Configures TensorFlow to use only the specified GPU indices.

            Args:
                gpu_indices (list of int): List of GPU indices to make visible to TensorFlow.
                                        Use an empty list to disable GPU usage.
            """
            gpus = tf.config.list_physical_devices('GPU')

            log = ctx.log
            log.debug("OH NO! I AM A TEST EXPERIMENT")

            if not gpus:
                log.warning("⚠️ No GPUs found.")
                return
            else:
                log.debug(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

            if gpu_indices:
                try:
                    selected_gpus = [gpus[i] for i in gpu_indices]
                    tf.config.set_visible_devices(selected_gpus, 'GPU')
                    for gpu in selected_gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    log.info(f"✅ Using GPU(s): {gpu_indices}")
                except IndexError:
                    raise ValueError(f"Invalid GPU index in {gpu_indices}. Available GPUs: {len(gpus)}")
                except RuntimeError as e:
                    print("❌ RuntimeError during GPU configuration:", e)
            else:
                try:
                    # Make no GPUs visible (force CPU mode)
                    tf.config.set_visible_devices([], 'GPU')
                    print("🚫 GPU usage disabled — running on CPU only.")
                except RuntimeError as e:
                    print("❌ RuntimeError while disabling GPU:", e)

        os.makedirs(ctx.experiment.save_dir, exist_ok=True)
        ctx.experiment.save_dir = os.path.abspath(ctx.experiment.save_dir)
        ctx.experiment.ckpt_file = os.path.join(ctx.experiment.save_dir, ctx.experiment.ckpt_file)

        seed = ctx.experiment.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

        configure_gpus(ctx.experiment.gpus)

