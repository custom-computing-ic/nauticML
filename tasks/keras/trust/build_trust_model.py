from tensorflow.keras.optimizers import Adam
from tasks.keras.trust.converter.dropout.mc_model import MonteCarloDropoutModel
from nautic import taskx

class KerasMCUncertaintyModel:
  @taskx
  def build_bayesian_model(ctx):
      model = ctx.model.logic
      if ctx.model.dropout_type == "mc":
          model = MonteCarloDropoutModel(model=model,
                                        nSamples=ctx.eval.mc_samples,
                                        p=ctx.model.dropout_rate,
                                        num=0,
                                        strategy="default",
                                        seed=ctx.experiment.seed,
                                        input=None)

      else:
        raise NotImplementedError("dropout type is not supportred")

      model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(learning_rate=ctx.train.learning_rate),
                  metrics=["accuracy"])

      ctx.model.logic = model
