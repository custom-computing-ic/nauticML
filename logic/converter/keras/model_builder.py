from tensorflow.keras.optimizers import Adam
from .dropout.mc_model import MonteCarloDropoutModel
from prefect import task
# needed for inference

@task
def build_uncertainty_model(cfg):
    model = cfg.model.logic
    if cfg.model.dropout_type == "mc":
        model = MonteCarloDropoutModel(model=model,
                                       nSamples=cfg.evaluation.mc_samples,
                                       p=cfg.model.dropout_rate,
                                       num=0,
                                       strategy="default",
                                       seed=cfg.output.seed,
                                       input=None)

    elif cfg.model.dropout_type == "mask":
      model = MasksemblesModel(model, num_masks=args.num_masks, scale=args.scale, num=0)
    else:
      raise NotImplementedError("dropout type is not supportred")

    model.compile(loss="categorical_crossentropy",
                optimizer=Adam(learning_rate=cfg.training.learning_rate), metrics=["accuracy"])

    return cfg