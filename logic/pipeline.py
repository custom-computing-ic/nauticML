from typing import Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
from .datasets import CIFAR10Data
from prefect.artifacts import create_table_artifact, \
                              create_progress_artifact,\
                              update_progress_artifact

from prefect import get_run_logger

from logic.models.lenet import LeNet
from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer
from logic.converter.keras.dropout.mc_model import MonteCarloDropoutModel
import tensorflow_probability as tfp

from prefect import task


@task(cache_policy=None)
def get_dataset(cfg):
    """
    Loads and preprocesses the dataset specified in args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        dict: A dictionary containing training, test, and validation datasets.
    """

    name = cfg.dataset.name
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
    elif cfg.dataset == "cifar10":
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

        print('num train:%d num val:%d' % (num_train, num_val))
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

    cfg.dataset.data = data
    return cfg


@task(cache_policy=None)
def get_model(cfg):
    """
    Constructs the model architecture based on the selected model and configuration.

    Args:
        args: Parsed command-line arguments.

    Returns:
        model: A compiled Keras model.
    """

    if cfg.model.name == "lenet":
        if cfg.model.is_quant:
            model = Qlenet(args, model_num_layer[args.model_name])
        else:
            model = LeNet(cfg)
    elif cfg.model.name == "resnet":
        if cfg.model.is_quant:
            model = QResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=64)
        else:
            model = ResNet18(input_shape=(32, 32, 3), classes=10, args=args, weight_decay=1e-4, base_filters=64)
    elif cfg.model.name == "vgg":
        if cfg.model.is_quant:
            model = QVGG11(args, filters=16, dense_out=[16, 16, 10])
        else:
            model = VGG11(args, filters=16, dense_out=[16, 16, 10])
    else:
        raise NotImplementedError("Model not supoorted")

    cfg.model.logic = model
    return cfg

@task(cache_policy=None)
def train_model(cfg):
    """
    Trains the model on the provided dataset using data augmentation, pruning, and custom learning rate schedulers.

    Args:
        args: Parsed command-line arguments.
        model: The Keras model to train.
        dataset: The dataset dictionary returned by get_dataset().
    """

    logger = get_run_logger()
    model = cfg.model.logic
    dataset = cfg.dataset.data



    if cfg.model.name == "lenet":
        chkp = ModelCheckpoint(
            cfg.output.ckpt_pathname,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            #save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        if cfg.model.p_rate != 0.0:
            callbacks = [chkp, pruning_callbacks.UpdatePruningStep() ]

        else:
            callbacks = [chkp]


        progress_artifact_id = create_progress_artifact(
            progress=0.0,
            description="Training!!!",
        )
        class PrintEpochCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                blah = round((epoch / cfg.training.num_epoch) * 100)
                print(f"blah: ", blah)
                update_progress_artifact(progress_artifact_id, progress=blah)


        callbacks.append(PrintEpochCallback())



        logger.info(f"p_rate: {cfg.model.p_rate}")
        logger.info(f"batch_size: {cfg.training.batch_size}")
        logger.info(f"num_epoch: {cfg.training.num_epoch}")
        logger.debug(f"validation_split: {cfg.training.validation_split}")


        def hash_model(model):
            import hashlib
            import json

            # Serialize model architecture to JSON string
            model_json = model.to_json()

            # Normalize JSON to remove whitespace and ensure consistent ordering
            model_dict = json.loads(model_json)

            normalized_json = json.dumps(model_dict, sort_keys=True)

            # Compute MD5 hash of the normalized string
            model_hash = hashlib.md5(normalized_json.encode('utf-8')).hexdigest()


            weight_hasher = hashlib.md5()

            for layer in model.layers:
                for w in layer.get_weights():
                    weight_hasher.update(np.array(w).tobytes())
            return (model_hash, weight_hasher.hexdigest())

        #print("Model hash:", hash_model(model))

        train_stat = model.fit(
            dataset['x_train'],
            dataset['y_train'],
            batch_size=cfg.training.batch_size,
            epochs=cfg.training.num_epoch,
            initial_epoch=1,
            validation_split=cfg.training.validation_split,
            callbacks=callbacks)

    elif args.model_name == "resnet":
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
        print('train with data augmentation')
        train_gen = datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size)
        # def lr_scheduler(epoch):
        #     lr = args.lr
        #     new_lr = lr
        #     if epoch <= 91:
        #         pass
        #     elif epoch > 91 and epoch <= 137:
        #         new_lr = lr * 0.1
        #     else:
        #         new_lr = lr * 0.01
        #     print('new lr:%.2e' % new_lr)
        #     return new_lr
        def lr_scheduler(epoch):
            lr = args.lr
            new_lr = lr * (0.1 ** (epoch // 50))
            print('new lr:%.2e' % new_lr)
            return new_lr
        reduce_lr = CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)

        chkp = ModelCheckpoint(
            ckpt_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            #save_weights_only=True,
            mode="auto",
            save_freq="epoch",
        )

        if args.p_rate != 0.0:
            callbacks = [reduce_lr, chkp,  pruning_callbacks.UpdatePruningStep() ]
        else:
            callbacks = [reduce_lr, chkp]

        history = model.fit_generator(generator=train_gen,
                                           epochs=args.num_epoch,
                                           callbacks=callbacks,
                                           validation_data=(dataset['x_val'], dataset['y_val']),
                                           )

    elif args.model_name == "vgg":
        callbacks = [CosineAnnealingScheduler(T_max=args.num_epoch, eta_max=args.lr, eta_min=1e-4)]
        datagen = ImageDataGenerator(rotation_range=8,
                                    zoom_range=[0.95, 1.05],
                                    height_shift_range=0.10,
                                    shear_range=0.15)
        train_stat = model.fit(datagen.flow(dataset['x_train'], dataset['y_train'], batch_size=args.batch_size),
            epochs=args.num_epoch, validation_data=(dataset['x_test'], dataset['y_test']), callbacks=callbacks)
    else:
        raise NotImplementedError("Training not supoorted")

    return cfg

@task
def eval_trust(cfg):
    """
    Loads the best model checkpoint and evaluates it using accuracy, entropy, and expected calibration error (ECE).

    Args:
        args: Parsed command-line arguments.
        model: The Keras model object (may be loaded again internally).
        dataset: The dataset dictionary.

    Returns:
        tuple: accuracy, ECE, and average predictive entropy (aPE)
    """

    def check_sparsity(model):
        allWeightsByLayer = {}

        print("\n")
        print("Checking Sparity")
        for layer in model.layers:
            if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
                continue
            weights=layer.weights[0].numpy().flatten()
            allWeightsByLayer[layer._name] = weights
            print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    # generate random noise dataset
    def random_noise_data(dataset_name):

        SVHN_mean = tuple([x / 255 for x in [129.3, 124.1, 112.4]])
        SVHN_std = tuple([x / 255 for x in [68.2, 65.4, 70.4]])
        MNIST_mean = (0,)
        MNIST_std = (1,)
        CIFAR10_mean = (0.4914, 0.4822, 0.4465)
        CIFAR10_std = (0.2023, 0.1994, 0.2010)

        if dataset_name == "mnist":
                # generate random noise test dataset with mean MNIST_mean and std MNIST_std
                x_test = np.random.normal(MNIST_mean, MNIST_std, [10000, 28, 28, 1])
                x_test = x_test.astype('float32')
                return x_test
        elif dataset_name == "cifar10":
                # generate random noise test dataset with mean CIFAR10_mean and std CIFAR10_std
                (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                x_test = np.random.normal(CIFAR10_mean, CIFAR10_std, x_test.shape)
                x_test = x_test.astype('float32')
                return x_test
        elif dataset_name ==  "svhn":
                # generate random noise test dataset with mean SVHN_mean and std SVHN_std
                x_test = np.random.normal(SVHN_mean, SVHN_std, [10000, 32, 32, 3])
                x_test = x_test.astype('float32')
                return x_test

    def entropy(output):
        batch_size = output.shape[0]
        entropy = -np.sum(np.log(output+1e-8)*output)/batch_size
        return entropy

    co = {  "BayesianDropout": InferenceDropoutLayer,
            "MCDropout": MonteCarloDropoutModel,
            # "Masksembles": Masksembles,
            # "MasksemblesModel": MasksemblesModel,
            "PruneLowMagnitude": pruning_wrapper.PruneLowMagnitude
            }
    _add_supported_quantized_objects(co)
    print("load model from:", cfg.output.ckpt_pathname)
    model = load_model(cfg.output.ckpt_pathname, custom_objects=co)
    model.summary()
    check_sparsity(model)
    data_dict = cfg.dataset.data

    #x_test_random = random_noise_data("mnist")[:args.num_eval_images]
    x_test_random_full = random_noise_data(cfg.dataset.name)

    from sklearn.metrics import accuracy_score

    y_prob        = model.predict(data_dict["x_test"])

    y_logits    = np.log(y_prob/(1-y_prob + 1e-15))
    ece_keras   = tfp.stats.expected_calibration_error(\
        num_bins=cfg.evaluation.num_bins,
        logits=y_logits,
        labels_true=np.argmax(data_dict["y_test"],axis=1),
        labels_predicted=np.argmax(y_prob,axis=1))

    accuracy_keras = float(accuracy_score(np.argmax(data_dict["y_test"],axis=1),
                                          np.argmax(y_prob,axis=1)))
    entropy_keras = entropy(model.predict(np.ascontiguousarray(x_test_random_full)))


    cfg.eval_trust.ece = float(ece_keras)
    cfg.eval_trust.accuracy = accuracy_keras
    cfg.eval_trust.ape = entropy_keras

    return cfg

@task
def prune_model(cfg):

    model = cfg.model.logic
    x_train_len = len(cfg.dataset.data['x_train'])
    if cfg.model.p_rate == 0.0:
        return cfg
    NSTEPS =   int(x_train_len)  // cfg.training.batch_size

    def pruneFunction(layer):
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=cfg.model.p_rate,
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

    model.compile(optimizer=SGD(learning_rate = cfg.training.learning_rate),
                  loss=['categorical_crossentropy'], metrics=['accuracy'])
    cfg.model.logic = model

    return cfg



@task
def compute_score(accuracy, ape, ece, flops: int, cfg) -> float:
    acc_base = {"lenet": 0.99, "resnet": 0.92}
    ape_base = {"lenet": 1.5, "resnet": 1.8}
    ece_base = {"lenet": 0.03, "resnet": 0.09}
    flops_base = {"lenet": 5340192, "resnet": 1112422588}

    weights = cfg.evaluation.w_list
    model_name = cfg.model.name

    final_score = float(accuracy / acc_base[model_name])   * cfg.evaluation.w_list[0] \
                + float(ape      / ape_base[model_name])   * cfg.evaluation.w_list[1] \
                - float(ece      / ece_base[model_name])   * cfg.evaluation.w_list[2] \
                - float(flops    / flops_base[model_name]) * cfg.evaluation.w_list[2]
    return final_score



