from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from rabbitcam_ml.transforms import PreprocessConfig, build_transform, letterbox_image


def test_inference_transform_has_expected_shape_and_replicated_grayscale() -> None:
    image = Image.fromarray(
        np.stack(
            [
                np.full((20, 40), 255, dtype=np.uint8),
                np.zeros((20, 40), dtype=np.uint8),
                np.zeros((20, 40), dtype=np.uint8),
            ],
            axis=2,
        )
    )
    transform = build_transform(
        {
            "input_size": [32, 32],
            "grayscale": True,
            "mean": [0, 0, 0],
            "std": [1, 1, 1],
            "pad_value": 0,
            "interpolation": "bilinear",
        }
    )
    output = transform(image)
    assert output.shape == (3, 32, 32)
    assert output.dtype == torch.float32
    assert torch.equal(output[0], output[1])
    assert torch.equal(output[1], output[2])
    # The 2:1 input is letterboxed with padding above and below.
    assert torch.count_nonzero(output[:, :8, :]) == 0
    assert torch.count_nonzero(output[:, 24:, :]) == 0


def test_letterbox_preserves_aspect_ratio() -> None:
    source = Image.new("L", (80, 20), color=255)
    result = letterbox_image(source, (40, 40), fill=0)
    array = np.asarray(result)
    nonzero_rows = np.flatnonzero(array.max(axis=1))
    assert result.size == (40, 40)
    assert len(nonzero_rows) == 10


def test_shipped_config_aliases_are_supported() -> None:
    config = PreprocessConfig.from_mapping(
        {
            "input_size": [64, 48],
            "letterbox": True,
            "pad_value": 12,
            "interpolation": "bilinear",
            "augmentation": {
                "max_rotation_degrees": 4,
                "max_translation_fraction": 0.03,
            },
        }
    )
    assert config.input_size == (64, 48)
    assert config.letterbox_fill == 12
    assert config.rotation_degrees == 4
    assert config.translation_fraction == pytest.approx(0.03)


def test_unknown_preprocessing_key_fails_clearly() -> None:
    with pytest.raises(ValueError, match="mystery"):
        build_transform({"mystery": True})
