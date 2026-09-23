"""Stable, reusable single-image inference API."""

from __future__ import annotations

import io
import json
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image

from .models.checkpoint import CheckpointError, load_checkpoint
from .models.factory import create_model

REQUIRED_LABELS = frozenset({"standing", "lateral", "unknown"})


@dataclass(frozen=True)
class Prediction:
    """One model decision, including unmodified class probabilities."""

    label: str
    raw_label: str
    confidence: float
    probabilities: dict[str, float]
    inference_time_ms: float
    checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""

    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def apply_confidence_threshold(
    raw_label: str,
    confidence: float,
    threshold: float | None,
) -> str:
    """Resolve a low-confidence argmax to the public ``unknown`` label."""

    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence threshold must be between 0 and 1")
    return "unknown" if threshold is not None and confidence < threshold else raw_label


def _checkpoint_threshold(checkpoint: Mapping[str, Any]) -> float | None:
    candidates = [
        checkpoint.get("inference_config"),
        checkpoint.get("metadata", {}).get("inference_config")
        if isinstance(checkpoint.get("metadata"), Mapping)
        else None,
        checkpoint.get("training_config", {}).get("inference")
        if isinstance(checkpoint.get("training_config"), Mapping)
        else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping) and candidate.get("confidence_threshold") is not None:
            return float(candidate["confidence_threshold"])
    metadata = checkpoint.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("confidence_threshold") is not None:
        return float(metadata["confidence_threshold"])
    return None


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise RuntimeError("Model output does not contain a logits tensor")


class Predictor:
    """Load a checkpoint once and perform repeated, stateless predictions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "auto",
        confidence_threshold: float | None = None,
        *,
        model_factory: Callable[..., torch.nn.Module] = create_model,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = resolve_device(device)
        checkpoint = load_checkpoint(self.checkpoint_path, map_location="cpu")

        class_to_idx = {str(key): int(value) for key, value in checkpoint["class_to_idx"].items()}
        if set(class_to_idx) != REQUIRED_LABELS:
            raise CheckpointError(
                "RabbitCam checkpoints must contain exactly standing, lateral, and unknown; "
                f"found {sorted(class_to_idx)}"
            )
        self.class_to_idx = class_to_idx
        self.idx_to_class = {index: label for label, index in class_to_idx.items()}
        expected_indices = set(range(len(class_to_idx)))
        if set(self.idx_to_class) != expected_indices:
            raise CheckpointError("Checkpoint class indices must be contiguous from zero")

        self.model_name = str(checkpoint["model_name"])
        metadata = checkpoint.get("metadata")
        model_kwargs = (
            dict(metadata.get("model_kwargs", {}))
            if isinstance(metadata, Mapping) and isinstance(metadata.get("model_kwargs", {}), Mapping)
            else {}
        )
        self.model = model_factory(
            model_name=self.model_name,
            num_classes=len(class_to_idx),
            pretrained=False,
            **model_kwargs,
        )
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except Exception as exc:
            raise CheckpointError(
                f"Model weights in {self.checkpoint_path} are incompatible with "
                f"architecture {self.model_name!r}: {exc}"
            ) from exc
        self.model.to(self.device)
        self.model.eval()

        self.preprocessing_config = dict(
            checkpoint.get("preprocessing_config", checkpoint.get("preprocessing", {}))
        )
        if transform is None:
            from .transforms import build_transform

            transform = build_transform(self.preprocessing_config, training=False)
        self.transform = transform

        stored_threshold = _checkpoint_threshold(checkpoint)
        self.confidence_threshold = (
            stored_threshold if confidence_threshold is None else float(confidence_threshold)
        )
        if self.confidence_threshold is not None and not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be between 0 and 1")
        self._lock = threading.Lock()

    def predict_pil(self, image: Image.Image) -> Prediction:
        """Predict from a Pillow image without reloading model state."""

        if not isinstance(image, Image.Image):
            raise TypeError("predict_pil expects a PIL.Image.Image")
        tensor = self.transform(image)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            shape = getattr(tensor, "shape", None)
            raise RuntimeError(f"Inference transform must return a CHW tensor, got {shape}")
        batch = tensor.unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            logits = _extract_logits(self.model(batch))
            if logits.ndim != 2 or logits.shape != (1, len(self.class_to_idx)):
                raise RuntimeError(
                    "Model must return logits shaped "
                    f"(1, {len(self.class_to_idx)}), got {tuple(logits.shape)}"
                )
            probabilities_tensor = torch.softmax(logits[0].float(), dim=0)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        scores = probabilities_tensor.detach().cpu().tolist()
        raw_index = int(np.argmax(scores))
        raw_label = self.idx_to_class[raw_index]
        confidence = float(scores[raw_index])
        label = apply_confidence_threshold(
            raw_label, confidence, self.confidence_threshold
        )
        probabilities = {
            self.idx_to_class[index]: float(scores[index]) for index in range(len(scores))
        }
        return Prediction(
            label=label,
            raw_label=raw_label,
            confidence=confidence,
            probabilities=probabilities,
            checkpoint=str(self.checkpoint_path),
            inference_time_ms=float(elapsed_ms),
        )

    def predict_bytes(self, data: bytes | bytearray | memoryview) -> Prediction:
        """Decode common image bytes and predict."""

        if not data:
            raise ValueError("Image bytes are empty")
        try:
            with Image.open(io.BytesIO(bytes(data))) as opened:
                image = opened.convert("RGB")
        except Exception as exc:
            raise ValueError("Could not decode image bytes") from exc
        return self.predict_pil(image)

    def predict_numpy(self, image: np.ndarray) -> Prediction:
        """Predict from a uint8 HxW, HxWx3 (RGB), or HxWx4 array."""

        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError(f"Expected a HxW or HxWxC image, got shape {array.shape}")
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {array.shape[2]}")
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Image contains non-finite values")
            if array.size and array.min() >= 0.0 and array.max() <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return self.predict_pil(Image.fromarray(array))

    def predict_path(self, image_path: str | Path) -> Prediction:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        return self.predict_pil(image)

    def predict_stream(self, stream_url: str, *, timeout: float = 10.0) -> Prediction:
        """Read one JPEG from an MJPEG URL and predict it."""

        return self.predict_bytes(read_one_mjpeg_frame(stream_url, timeout=timeout))


def read_one_mjpeg_frame(
    stream_url: str,
    *,
    timeout: float = 10.0,
    chunk_size: int = 16_384,
    max_frame_bytes: int = 20_000_000,
) -> bytes:
    """Read the first complete JPEG frame from an MJPEG response."""

    if not stream_url:
        raise ValueError("stream_url must be non-empty")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    started = time.monotonic()
    buffer = bytearray()
    try:
        with urllib.request.urlopen(stream_url, timeout=timeout) as response:
            while time.monotonic() - started < timeout:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                buffer.extend(chunk)
                start = buffer.find(b"\xff\xd8")
                if start >= 0:
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end >= 0:
                        return bytes(buffer[start : end + 2])
                    if start:
                        del buffer[:start]
                elif len(buffer) > chunk_size * 2:
                    del buffer[:-2]
                if len(buffer) > max_frame_bytes:
                    raise RuntimeError("MJPEG frame exceeds the configured size limit")
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not read MJPEG stream {stream_url!r}: {exc}") from exc
    raise RuntimeError(f"No complete JPEG frame received from {stream_url!r}")


def format_prediction(prediction: Prediction) -> str:
    """Format the concise human-readable CLI output."""

    lines = [
        f"prediction: {prediction.label}",
        f"raw prediction: {prediction.raw_label}",
        f"confidence: {prediction.confidence:.3f}",
        "",
    ]
    lines.extend(
        f"{label}: {probability:.3f}"
        for label, probability in prediction.probabilities.items()
    )
    return "\n".join(lines)


__all__ = [
    "Prediction",
    "Predictor",
    "apply_confidence_threshold",
    "format_prediction",
    "read_one_mjpeg_frame",
    "resolve_device",
]
