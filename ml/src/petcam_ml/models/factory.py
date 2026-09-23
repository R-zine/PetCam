"""Central model factory.

Keeping model creation here makes checkpoints and inference independent from the
training loop.  The tiny model is intentionally part of the public factory: it
provides an offline, fast architecture for tests and pipeline smoke checks.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

DEFAULT_MODEL_NAME = "mobilenetv4_conv_small.e3600_r256_in1k"
TINY_MODEL_NAMES = frozenset({"tiny", "tiny_cnn", "test_tiny"})


class TinyClassifier(nn.Module):
    """A small convolutional classifier intended for tests, not deployment."""

    def __init__(self, num_classes: int = 3, width: int = 8) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if width < 1:
            raise ValueError("width must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def create_model(
    model_name: str = DEFAULT_MODEL_NAME,
    num_classes: int = 3,
    pretrained: bool = True,
    **model_kwargs: Any,
) -> nn.Module:
    """Create a classifier with ``num_classes`` outputs.

    ``tiny_cnn`` (plus the aliases in :data:`TINY_MODEL_NAMES`) never imports
    timm or accesses the network, which keeps unit tests fully offline.
    """

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    normalized_name = model_name.strip().lower()
    if normalized_name in TINY_MODEL_NAMES:
        # ``pretrained`` deliberately has no effect for this test architecture.
        width = int(model_kwargs.pop("width", 8))
        if model_kwargs:
            unexpected = ", ".join(sorted(model_kwargs))
            raise TypeError(f"Unsupported tiny model options: {unexpected}")
        return TinyClassifier(num_classes=num_classes, width=width)

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "timm is required for production backbones. Install petcam-ml "
            "with its core dependencies, or use model_name='tiny_cnn' in tests."
        ) from exc

    try:
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            **model_kwargs,
        )
    except Exception as exc:
        weight_hint = (
            " Pretrained weights may require network access; use pretrained=False "
            "for an offline smoke test."
            if pretrained
            else ""
        )
        raise RuntimeError(f"Could not create model {model_name!r}.{weight_hint}") from exc


# A readable alias for callers that prefer build-style factory names.
build_model = create_model

__all__ = [
    "DEFAULT_MODEL_NAME",
    "TINY_MODEL_NAMES",
    "TinyClassifier",
    "build_model",
    "create_model",
]
