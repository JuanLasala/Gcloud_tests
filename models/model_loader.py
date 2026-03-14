import torch
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from transformers import AutoModelForImageClassification
from transformers.modeling_outputs import ImageClassifierOutput


def _find_first_conv(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            return name, module
    raise ValueError("No Conv2d layer found in model.")


def _replace_module(root: torch.nn.Module, name: str, new_module: torch.nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _expand_first_conv(model, in_channels):
    _, old_conv = _find_first_conv(model)
    new_conv = torch.nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        if in_channels >= old_conv.in_channels:
            new_conv.weight[:, : old_conv.in_channels, :, :] = old_conv.weight
            if in_channels > old_conv.in_channels:
                src_mean = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.weight[:, old_conv.in_channels :, :, :] = src_mean.repeat(
                    1, in_channels - old_conv.in_channels, 1, 1
                )
        else:
            new_conv.weight.copy_(old_conv.weight[:, :in_channels, :, :])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    first_conv_name, _ = _find_first_conv(model)
    _replace_module(model, first_conv_name, new_conv)


class TorchvisionEfficientNetForClassification(torch.nn.Module):
    def __init__(self, num_labels, id2label, label2id, in_channels, pretrained=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        self.efficientnet = efficientnet_v2_s(weights=weights)
        if in_channels != 3:
            _expand_first_conv(self.efficientnet, in_channels)

        in_features = self.efficientnet.classifier[-1].in_features
        self.efficientnet.classifier[-1] = torch.nn.Linear(in_features, num_labels)

        self.config = type("Config", (), {})()
        self.config.num_labels = num_labels
        self.config.id2label = id2label
        self.config.label2id = label2id
        self.config.num_channels = in_channels

    def forward(self, pixel_values=None, labels=None, **kwargs):
        logits = self.efficientnet(pixel_values)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return ImageClassifierOutput(loss=loss, logits=logits)


def load_hf_model(model_name, num_labels, id2label, label2id, in_channels=12, use_rgb=False):
    if model_name == "torchvision/efficientnet_v2_s":
        model = TorchvisionEfficientNetForClassification(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            in_channels=3 if use_rgb else in_channels,
            pretrained=True,
        )
        return model, None

    # Some EfficientNet IDs used online are aliases/typos and don't exist on HF Hub.
    # Try requested name first, then known public fallbacks.
    candidates = [model_name]
    if model_name == "google/efficientnet-v2-s":
        candidates.extend(
            [
                "google/efficientnet-b4",
            ]
        )

    model = None
    last_error = None
    for candidate in candidates:
        try:
            model = AutoModelForImageClassification.from_pretrained(
                candidate,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            if candidate != model_name:
                print(f"[WARN] Model '{model_name}' not found. Using fallback '{candidate}'.")
            break
        except OSError as err:
            last_error = err

    if model is None:
        raise OSError(
            f"Unable to load model '{model_name}'. Tried: {candidates}. "
            "Check repo ID and Hugging Face authentication."
        ) from last_error

    if hasattr(model.config, "num_channels"):
        model.config.num_channels = 3 if use_rgb else in_channels

    if not use_rgb:
        _expand_first_conv(model, in_channels)
    return model, None
