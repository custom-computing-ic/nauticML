
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from logic.datasets import CIFAR10Data
from nautic import taskx

class KerasDataset:
    @taskx
    def get_dataset(ctx):
        """
        Loads and preprocesses the dataset specified in args.

        Args:
            args: Parsed command-line arguments.

        Returns:
            dict: A dictionary containing training, test, and validation datasets.
        """

        name = ctx.dataset.name
        if name == "mnist":
            num_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            RESHAPED = 784

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")

            x_train = x_train[..., np.newaxis]
            x_test = x_test[..., np.newaxis]

            x_train /= 256.0
            x_test /= 256.0
            y_train = to_categorical(y_train, num_classes)
            y_test = to_categorical(y_test, num_classes)
        elif name == "cifar10":
            cifar10_data = CIFAR10Data()
            x_train, y_train, x_test, y_test = cifar10_data.get_data(subtract_mean=True)

            num_train = int(x_train.shape[0] * 0.9)
            num_val = x_train.shape[0] - num_train
            mask = list(range(num_train, num_train+num_val))
            x_val = x_train[mask]
            y_val = y_train[mask]

            mask = list(range(num_train))
            x_train = x_train[mask]
            y_train = y_train[mask]

            data = (x_train, y_train, x_val, y_val, x_test, y_test)
        else:
            raise NotImplementedError("Dataset not supoorted")

        if name == "cifar10":
            data = {"x_train": x_train,
                    "x_test": x_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "x_val": x_val,
                    "y_val": y_val}
        else:
            data = {"x_train": x_train,
                    "x_test": x_test,
                    "y_train": y_train,
                    "y_test": y_test}

        ctx.dataset.data = data
