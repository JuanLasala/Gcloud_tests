def _get_model_input_channels(model, default=13):
    if hasattr(model, "config") and hasattr(model.config, "num_channels"):
        return int(model.config.num_channels)
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            return int(module.in_channels)
    return default