from tensorflow.keras.models import Sequential
from tensorflow.keras import activations
from keras.layers import Dense, Activation, \
                         Flatten, \
                         Conv2D, MaxPool2D

from keras.optimizers import SGD

from tasks.keras.trust.converter.uncertainty_factory import get_uncertainty_layer

def LeNet(cfg):

    num_layers = 3

    if cfg.model.num_bayes_layer not in range(0, num_layers + 1):
        raise ValueError("num_bayes_layer must be in range [0, {}]".format(num_layers))

    model=Sequential()

    num_nonbayes_layer = num_layers - cfg.model.num_bayes_layer - 1
    # Lenet
    # Convolutional layer

    model.add(Conv2D(filters=int(20*cfg.model.scale_factor), kernel_size=(5,5), \
                     input_shape=(28,28,1), padding = "same", name="conv2d_1"))
    model.add(Activation(activations.relu, name='relu1'))
    # Max-pooing layer with pooling window size is 2x2
    model.add(MaxPool2D(pool_size=(2,2), strides=2))

    # MC dropout
    if (num_nonbayes_layer < 0):
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # Convolutional layer
    model.add(Conv2D(filters=int(20*cfg.model.scale_factor), kernel_size=(5,5), \
                     padding="same", name="conv2d_2"))
    model.add(Activation(activations.relu, name='relu2'))
    # Max-pooling layer
    model.add(MaxPool2D(pool_size=(7,7), strides=7))

    # Flatten layer
    model.add(Flatten())

    # MC dropout
    if (num_nonbayes_layer < 0):
        model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    # The first fully connected layer
    model.add(Dense(int(100*cfg.model.scale_factor), name="fc_1"))
    model.add(Activation(activations.relu, name='relu3'))
    # The output layer

    # MC dropout
    if (num_nonbayes_layer < 0): model.add(get_uncertainty_layer(cfg))
    num_nonbayes_layer -= 1

    model.add(Dense(10, name="fc_2"))
    model.add(Activation(activation='softmax', name='softmax'))
    model.compile(optimizer=SGD(learning_rate = cfg.train.learning_rate), \
                                loss=['categorical_crossentropy'],\
                                metrics=['accuracy'])
    return model