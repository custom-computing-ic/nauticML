
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
        elif name == "jets":
            # Canonical hls4ml jet-tagging dataset, faithful to the tutorial:
            # OpenML 'hls4ml_lhc_jets_hlf' (830k jets, 16 high-level features,
            # 5 classes), LabelEncoder + one-hot, 80/20 split, StandardScaler
            # fit on train only.
            from sklearn.datasets import fetch_openml
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder, StandardScaler

            num_classes = 5
            data_bunch = fetch_openml("hls4ml_lhc_jets_hlf", version=1, as_frame=False)
            X = np.asarray(data_bunch["data"], dtype="float32")
            y = LabelEncoder().fit_transform(data_bunch["target"])
            y = to_categorical(y, num_classes)

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=ctx.experiment.seed)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train).astype("float32")
            x_test = scaler.transform(x_test).astype("float32")

        elif name == "tracking":
            # wa-hls4ml "Particle Tracking" model (14 features, 3 classes).
            # The ORIGINAL dataset is not publicly available, so we train on a
            # reproducible synthetic stand-in with the same input/output spec.
            # >>> Swap this block for the real data when available; nothing
            # downstream depends on how these arrays are produced. <<<
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            num_classes = 3
            X, y = make_classification(
                n_samples=60000, n_features=14, n_informative=10,
                n_redundant=2, n_classes=num_classes,
                random_state=ctx.experiment.seed)
            X = X.astype("float32")
            y = to_categorical(y, num_classes)

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=ctx.experiment.seed)

            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train).astype("float32")
            x_test = scaler.transform(x_test).astype("float32")

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
