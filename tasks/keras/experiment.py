import os
import shutil
from typing import List

os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Force TF deterministic ops
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # CuDNN determinism for certain layers
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'  # Disables AMP which can introduce nondeterminism

import tensorflow as tf

import random
import numpy as np
from nautic import taskx

class KerasExperiment:

    @taskx
    def initialize_experiment(ctx):
        def configure_gpus(use_cpu: bool, gpu_indices : List[int] | None):
            """
            Configures TensorFlow to use only the specified GPU indices.

            Args:
                gpu_indices (list of int): List of GPU indices to make visible to TensorFlow.
                                        Use an empty list to disable GPU usage.
            """

            log = ctx.log
            log.debug("I AM A TEST EXPERIMENT - In DEBUG Mode")

            if use_cpu:
                try:
                    # Make no GPUs visible (force CPU mode)
                    tf.config.set_visible_devices([], 'GPU')
                    log.info("🚫 GPU usage disabled — running on CPU only.")
                except RuntimeError as e:
                    log.info("❌ RuntimeError while disabling GPU:", e)
                finally:
                    return

            gpus = tf.config.list_physical_devices('GPU')

            if not gpus:
                log.warning("⚠️ No GPUs found.")
                return
            else:
                log.debug(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")

            if gpu_indices and len(gpu_indices) > 0:
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
                log.warning("⚠️ CPU used by default as no CPU or GPU indices provided are empty")
                return configure_gpus(True, [])

        save_dir = ctx.experiment.save_dir

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.makedirs(save_dir)
        ctx.experiment.save_dir = os.path.abspath(save_dir)
        ctx.experiment.ckpt_file = os.path.join(save_dir, ctx.experiment.ckpt_file)

        seed = ctx.experiment.seed

        os.environ['PYTHONHASHSEED'] = str(seed)

        random.seed(seed)
        np.random.seed(seed)

        tf.keras.utils.set_random_seed(seed)
        tf.random.set_seed(seed)

        use_cpu = getattr(ctx.experiment, "cpu", False)
        configure_gpus(use_cpu, getattr(ctx.experiment, "gpus", []))
