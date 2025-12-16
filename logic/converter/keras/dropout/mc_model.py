from logic.converter.keras.nn2bnn import HlsLayer, _convert_model, strategy_fn
from .inference_layer import InferenceDropoutLayer
import tensorflow as tf


class MonteCarloDropoutModel(HlsLayer):
  r"""
   This class uses Morte Carlo Dropout to convert a traditional neural network to a Bayesian neural network.
   The mathematical proof is in this paper: https://arxiv.org/pdf/1506.02142.pdf. Note that the model to be
   converted should be contructed using either Sequential or Functional APIs.
  """

  def __init__(self, *, model, nSamples, p, num,
      strategy, seed, input, **kwargs):
      super().__init__()
      self.original_model = model
      supported_layers = strategy_fn[strategy](model, InferenceDropoutLayer, **kwargs)
      print(f"Converting model to BayesianDropout: {nSamples}, {p}, {num}, {strategy}, {seed}, {input}, {kwargs}")

      if num > 0:
        self.model = _convert_model(model, 'BayesianDropout', supported_layers, p, seed, input)
      else:
        self.model = model

      self.nSamples = nSamples
      self.p = p
      self.seed = seed

  def call(self, input, training=True):
      if training:
        return self.model(input, training=True)
      else:
        prediction = self.model(input, training=False)
        if isinstance(prediction, list):
          prediction = [prediction[i] for i in range(len(prediction))]
        else:
          pred_shape = prediction.shape
          if len(pred_shape) == 2: return prediction # No MC samples
          prediction = [prediction[i] for i in range(pred_shape[0])]
        return sum(prediction) / len(prediction)

  def get_config(self):
    config = super(MonteCarloDropoutModel, self).get_config()
    config["model"] = self.model
    config["nSamples"] = self.nSamples
    config["p"] = self.p
    config["seed"] = self.seed
    return config