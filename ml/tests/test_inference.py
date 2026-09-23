from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from rabbitcam_ml.inference import (
    Predictor,
    apply_confidence_threshold,
    read_one_mjpeg_frame,
)
from rabbitcam_ml.models.checkpoint import save_checkpoint
from rabbitcam_ml.models.factory import create_model


CLASS_TO_IDX = {"standing": 0, "lateral": 1, "unknown": 2}


def _tensor_transform(image: Image.Image) -> torch.Tensor:
    gray = np.asarray(image.convert("L").resize((16, 16)), dtype=np.float32) / 255.0
    return torch.from_numpy(np.repeat(gray[None, :, :], 3, axis=0))


def _constant_checkpoint(path: Path) -> Path:
    model = create_model("tiny_cnn", pretrained=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([0.1, 1.0, 0.0]))
    return save_checkpoint(
        path,
        model=model,
        epoch=0,
        global_step=1,
        best_metric=0.0,
        class_to_idx=CLASS_TO_IDX,
        model_name="tiny_cnn",
        preprocessing_config={"image_size": 16},
        training_config={"inference": {"confidence_threshold": 0.8}},
    )


def test_prediction_schema_and_confidence_fallback(tmp_path: Path) -> None:
    checkpoint = _constant_checkpoint(tmp_path / "model.pt")
    predictor = Predictor(checkpoint, device="cpu", transform=_tensor_transform)

    prediction = predictor.predict_numpy(np.zeros((20, 30, 3), dtype=np.uint8))
    payload = prediction.to_dict()

    assert payload.keys() == {
        "label",
        "raw_label",
        "confidence",
        "probabilities",
        "inference_time_ms",
        "checkpoint",
    }
    assert prediction.raw_label == "lateral"
    assert prediction.label == "unknown"
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
    assert set(prediction.probabilities) == set(CLASS_TO_IDX)
    assert prediction.inference_time_ms >= 0

    without_fallback = Predictor(
        checkpoint,
        device="cpu",
        confidence_threshold=0.0,
        transform=_tensor_transform,
    ).predict_pil(Image.new("RGB", (10, 10)))
    assert without_fallback.label == "lateral"
    assert apply_confidence_threshold("standing", 0.54, 0.55) == "unknown"
    assert apply_confidence_threshold("standing", 0.55, 0.55) == "standing"


def test_predict_bytes_rejects_bad_data_and_decodes_jpeg(tmp_path: Path) -> None:
    checkpoint = _constant_checkpoint(tmp_path / "model.pt")
    predictor = Predictor(checkpoint, device="cpu", transform=_tensor_transform)
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(encoded, format="JPEG")

    assert predictor.predict_bytes(encoded.getvalue()).raw_label == "lateral"
    with pytest.raises(ValueError, match="decode"):
        predictor.predict_bytes(b"not an image")


def test_read_one_mjpeg_frame_preserves_original_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    jpeg = b"\xff\xd8original jpeg bytes\xff\xd9"

    class Response:
        def __init__(self) -> None:
            self.parts = iter([b"headers and noise", jpeg[:8], jpeg[8:] + b"trailer", b""])

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return next(self.parts)

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    assert read_one_mjpeg_frame("http://camera:81/stream") == jpeg
