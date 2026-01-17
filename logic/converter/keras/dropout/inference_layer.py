import tensorflow as tf

class InferenceDropoutLayer(tf.keras.layers.Layer):
    r"""
        Applies Dropout to the input.
    """
    def __init__(self, drop_rate, seed, **kwargs):
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                                    "but got {}".format(drop_rate))
        super(InferenceDropoutLayer, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        # if seed set to None, random output will be given
        self.seed = seed
        # self.dropout_layer = tf.keras.layers.Dropout(rate=self.drop_rate, seed=self.seed)


    def get_config(self):
        # default seed is 0
        seed = 0 if self.seed is None else self.seed
        config = {
        'drop_rate': self.drop_rate,
        'seed': seed
        }
        base_config = super(InferenceDropoutLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input):
        return tf.keras.layers.Dropout(rate=self.drop_rate, seed=self.seed)(input, training=True)
