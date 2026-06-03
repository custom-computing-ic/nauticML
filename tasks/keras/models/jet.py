from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from tasks.keras.trust.converter.uncertainty_factory import get_uncertainty_layer


def Jet(cfg):
    """Canonical hls4ml jet-tagging MLP (Duarte et al. 2018 / Fahim et al. 2021).

    Architecture, faithful to the hls4ml tutorial (part 1)::

        Input(16) -> Dense(64) -> ReLU -> Dense(32) -> ReLU
                  -> Dense(32) -> ReLU -> Dense(5)  -> softmax

    ~4,389 parameters at scale_factor=1.0. Trained on the OpenML
    ``hls4ml_lhc_jets_hlf`` dataset: 16 high-level jet features, 5 classes
    (gluon, light quark, W, Z, top). Weight init ``lecun_uniform`` and an
    L1(1e-4) penalty match the reference model.

    Monte-Carlo-dropout Bayesian layers are inserted *after* each hidden ReLU,
    gated by ``cfg.model.num_bayes_layer`` using the same countdown trick as
    LeNet: with num_bayes_layer = k, the last k of the 3 insertion points get a
    BayesianDropout layer (k=3 -> all three hidden layers, k=1 -> only the
    layer feeding the softmax). This is the placement assumed by
    build_bayesian_model() / MonteCarloDropoutModel downstream.
    """

    num_layers = 3  # three hidden Dense layers => three dropout insertion points
    if cfg.model.num_bayes_layer not in range(0, num_layers + 1):
        raise ValueError("num_bayes_layer must be in range [0, {}]".format(num_layers))

    # scale_factor lets the DSE shrink/grow the net while keeping the 64:32:32
    # ratio of the canonical model. scale_factor=1.0 reproduces the paper exactly.
    h1 = max(1, int(64 * cfg.model.scale_factor))
    h2 = max(1, int(32 * cfg.model.scale_factor))
    h3 = max(1, int(32 * cfg.model.scale_factor))

    init = "lecun_uniform"
    reg = l1(0.0001)

    # Same countdown as LeNet: when it goes negative we drop a BayesianDropout
    # layer in, so dropout is added from the output backwards.
    num_nonbayes_layer = num_layers - cfg.model.num_bayes_layer - 1

    model = Sequential()

    # --- Hidden layer 1 ---
    model.add(Dense(h1, input_shape=(16,), name="fc1",
                    kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation(activations.relu, name="relu1"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Hidden layer 2 ---
    model.add(Dense(h2, name="fc2",
                    kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation(activations.relu, name="relu2"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Hidden layer 3 ---
    model.add(Dense(h3, name="fc3",
                    kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation(activations.relu, name="relu3"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Output layer ---
    # Named "output" so model_factory.prune() leaves it unpruned (it only
    # excludes the classification head, matching LeNet's "fc_2").
    model.add(Dense(5, name="output",
                    kernel_initializer=init, kernel_regularizer=reg))
    model.add(Activation(activation="softmax", name="softmax"))

    # Faithful to the tutorial: Adam @ 1e-4, categorical cross-entropy. Note the
    # framework may re-compile (build_bayesian_model uses Adam; prune() uses SGD),
    # exactly as it does for LeNet/ResNet.
    model.compile(optimizer=Adam(learning_rate=cfg.train.learning_rate),
                  loss=["categorical_crossentropy"],
                  metrics=["accuracy"])
    return model
