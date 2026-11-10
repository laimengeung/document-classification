"""Build any of the three compared models and split their parameters for the optimizer.

    custom_cnn       M1  from scratch                        (src/models/custom_cnn.py)
    resnet50         M2  torchvision, ImageNet weights       (IMAGENET1K_V2)
    efficientnet_b0  M3  timm, ImageNet weights

For the pretrained models only the final classifier is replaced (1000 -> 16
outputs). The fine-tuning strategy is **differential learning rates**: the
backbone trains at ``train.lr`` and the new head at ``train.head_lr``, so the
random head can learn quickly without wrecking the pretrained features.
"""

from __future__ import annotations

from torch import nn

from src.models.custom_cnn import CustomCNN

MODEL_NAMES = ("custom_cnn", "resnet50", "efficientnet_b0")


def build_model(model_cfg: dict) -> nn.Module:
    """Create the model described by the ``model`` section of a config."""
    name = model_cfg["name"]
    num_classes = model_cfg["num_classes"]
    if name == "custom_cnn":
        return CustomCNN(num_classes, model_cfg["in_chans"], model_cfg.get("dropout", 0.3))
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if model_cfg["pretrained"] else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        import timm

        return timm.create_model("efficientnet_b0", pretrained=model_cfg["pretrained"], num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}")


def get_head(model: nn.Module, name: str) -> nn.Module:
    """The final classifier: the part trained at ``head_lr``."""
    if name == "custom_cnn":
        return model.classifier
    if name == "resnet50":
        return model.fc
    if name == "efficientnet_b0":
        return model.get_classifier()
    raise ValueError(f"Unknown model {name!r}")


def get_param_groups(model: nn.Module, name: str, lr: float, head_lr: float | None,
                     weight_decay: float) -> list[dict]:
    """AdamW parameter groups: (backbone | head) x (decay | no decay).

    Biases and normalization weights (1-D tensors) get no weight decay, which
    is standard practice: decaying them only hurts.
    """
    head_ids = {id(p) for p in get_head(model, name).parameters()}
    groups = {key: [] for key in ("backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay")}
    for param in model.parameters():
        if not param.requires_grad:
            continue
        part = "head" if id(param) in head_ids else "backbone"
        kind = "no_decay" if param.ndim <= 1 else "decay"
        groups[f"{part}_{kind}"].append(param)
    head_lr = lr if head_lr is None else head_lr
    return [
        {"name": key, "params": params, "lr": head_lr if key.startswith("head") else lr,
         "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay}
        for key, params in groups.items() if params
    ]


def count_parameters(model: nn.Module) -> int:
    """Total number of parameters."""
    return sum(p.numel() for p in model.parameters())
