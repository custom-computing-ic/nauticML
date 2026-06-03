from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from tasks.keras.trust.converter.uncertainty_factory import get_uncertainty_layer


def ParticleTracking(cfg):
    """wa-hls4ml "Particle Tracking" MLP (Table 3 of arXiv:2511.05615).

    Architecture, faithful to the benchmark::

        Input(14) -> Dense(32) -> ReLU -> Dense(32) -> ReLU
                  -> Dense(32) -> ReLU -> Dense(3)  -> softmax

    ~2,691 parameters at scale_factor=1.0; a 3-class softmax classifier that
    drops straight into the accuracy / ECE / APE eval pipeline.

    DATA CAVEAT: the original dataset for this model is not publicly available
    (only the jet dataset is). The architecture and training recipe here are
    faithful, but it is trained on a reproducible synthetic stand-in produced in
    KerasDataset.get_dataset() ("tracking" branch). Swap that block for the real
    data when available — this builder does not care how the arrays were made.

    MC-dropout Bayesian layers are inserted after each hidden ReLU, gated by
    cfg.model.num_bayes_layer with the same countdown trick as LeNet/Jet.
    """

    num_layers = 3  # three hidden Dense layers => three dropout insertion points
    if cfg.model.num_bayes_layer not in range(0, num_layers + 1):
        raise ValueError("num_bayes_layer must be in range [0, {}]".format(num_layers))

    # scale_factor keeps the uniform 32-wide hidden shape while letting the DSE
    # resize the net; scale_factor=1.0 reproduces the benchmark architecture.
    h = max(1, int(32 * cfg.model.scale_factor))

    num_nonbayes_layer = num_layers - cfg.model.num_bayes_layer - 1

    model = Sequential()

    # --- Hidden layer 1 ---
    model.add(Dense(h, input_shape=(14,), name="fc1"))
    model.add(Activation(activations.relu, name="relu1"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Hidden layer 2 ---
    model.add(Dense(h, name="fc2"))
    model.add(Activation(activations.relu, name="relu2"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Hidden layer 3 ---
    model.add(Dense(h, name="fc3"))
    model.add(Activation(activations.relu, name="relu3"))
    if num_nonbayes_layer < 0:
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # --- Output layer ---
    # Named "output" so model_factory.prune() leaves the head unpruned.
    model.add(Dense(3, name="output"))
    model.add(Activation(activation="softmax", name="softmax"))

    model.compile(optimizer=Adam(learning_rate=cfg.train.learning_rate),
                  loss=["categorical_crossentropy"],
                  metrics=["accuracy"])
    return model
