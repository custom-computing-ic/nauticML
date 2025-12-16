from logic.converter.keras.dropout.inference_layer import InferenceDropoutLayer
#from logic.converter.keras.masksembles import MaskedEnsembleLayer

def insert_uncertainty_layer(cfg, input_tensor):
    """Wraps an input tensor with the appropriate uncertainty layer based on config."""
    if cfg.model.dropout_type == "mc":
        return InferenceDropoutLayer(cfg.model.dropout_rate)(input_tensor)
    elif cfg.model.dropout_type == "mask":
        return MaskedEnsembleLayer(n=cfg.model.num_masks, scale=cfg.model.scale)(input_tensor)
    else:
        raise NotImplementedError(f"Unsupported dropout type: {cfg.model.dropout_type}")

def get_uncertainty_layer(cfg):
    """Returns the appropriate uncertainty-aware layer class instance based on config."""
    if cfg.model.dropout_type == "mc":
        return InferenceDropoutLayer(drop_rate=cfg.model.dropout_rate, seed=cfg.experiment.seed)
    elif cfg.model.dropout_type == "mask":
        return MaskedEnsembleLayer(n=cfg.model.num_masks, scale=cfg.model.scale)
    else:
        raise NotImplementedError(f"Unsupported dropout type: {cfg.model.dropout_type}")
