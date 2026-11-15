"""Image transforms: resize, document-appropriate augmentation, channel handling, normalization.

Augmentations simulate scan variation only (PRD 5.2): small rotation, slight
scale/translate, brightness/contrast jitter, blur and noise. There are
deliberately **no flips and no large rotations**: mirrored or upside-down
pages do not occur and would destroy the layout cues the model should learn.
"""

from __future__ import annotations

import torch
from torchvision import transforms as T

PAPER_WHITE = 255  # fill colour for pixels exposed by rotation/translation


class AddGaussianNoise:
    """Add pixel noise with standard deviation ``std`` to a [0, 1] tensor, then clip."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor + torch.randn_like(tensor) * self.std).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class RepeatChannels:
    """Replicate a 1-channel tensor to ``n`` channels (grayscale -> RGB for ImageNet models)."""

    def __init__(self, n: int):
        self.n = n

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.expand(self.n, -1, -1) if self.n > 1 else tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={self.n})"


def build_transforms(cfg: dict, mean: list[float], std: list[float], train: bool) -> T.Compose:
    """Build the transform pipeline for grayscale PIL images.

    Args:
        cfg: full experiment config (uses ``data``, ``augment`` and ``model.in_chans``).
        mean, std: per-channel normalization statistics (length ``in_chans``).
        train: if True and ``augment.enabled``, add random augmentations.
    """
    size = cfg["data"]["image_size"]
    in_chans = cfg["model"]["in_chans"]
    aug = cfg["augment"]

    # Resize first so the random ops below run on the smaller image (faster).
    steps: list = [T.Resize((size, size), antialias=True)]
    if train and aug["enabled"]:
        steps += [
            T.RandomAffine(
                degrees=aug["rotation_deg"],
                translate=(aug["translate"], aug["translate"]),
                scale=(1 - aug["scale"], 1 + aug["scale"]),
                fill=PAPER_WHITE,
            ),
            T.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=aug["blur_p"]),
        ]
    steps.append(T.ToTensor())  # PIL "L" -> float tensor [1, H, W] in [0, 1]
    if train and aug["enabled"] and aug["noise_p"] > 0:
        steps.append(T.RandomApply([AddGaussianNoise(aug["noise_std"])], p=aug["noise_p"]))
    steps += [RepeatChannels(in_chans), T.Normalize(mean, std)]
    return T.Compose(steps)
