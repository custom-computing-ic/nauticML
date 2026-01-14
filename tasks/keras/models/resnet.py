from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D
from keras.regularizers import l2
from keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tasks.keras.trust.converter.dropout.mc_model import BayesianDropout

##############################  ResNet   ####################################
# Get from https://github.com/jerett/Keras-CIFAR10
def conv2d_bn(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer

def Insert_Bayesian_Layer(cfg, x):
  if cfg.model.dropout_type == "mc": x = BayesianDropout(cfg.model.dropout_rate)(x)
  else: raise NotImplementedError("dropout type is not supportred")
  return x

def conv2d_bn_relu(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('relu')(layer)
    return layer

def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out

def ResNet18(cfg, input_shape=(32, 32, 3), classes=10, num_bayes_loc=8, weight_decay=1e-4, base_filters=64):
    new_base_filters = max(1, int(base_filters * cfg.model.scale_factor))
    input = Input(shape=input_shape)
    x = input
    num_nonbayes_layer = num_bayes_loc - cfg.model.num_bayes_layer - 1
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu(x, filters=new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    # # conv 3
    x = ResidualBlock(x, filters=2*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=2*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    # # conv 4
    x = ResidualBlock(x, filters=4*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=4*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    # # conv 5
    x = ResidualBlock(x, filters=8*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1

    x = ResidualBlock(x, filters=8*new_base_filters, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # Insert Bayeian Layer, can be mc dropout or mask ensumble layer
    if (num_nonbayes_layer < 0): x = Insert_Bayesian_Layer(cfg, x)
    num_nonbayes_layer -= 1
    
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name='ResNet18')
    model.compile(optimizer=SGD(lr=cfg.train.learning_rate, momentum=0.9, nesterov=False), loss=['categorical_crossentropy'], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=args.lr, amsgrad=True), loss=['categorical_crossentropy'], metrics=['accuracy'])
    return model