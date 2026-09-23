from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image
import pytest

from rabbitcam_ml.camera.mjpeg import JPEGFrame
from rabbitcam_ml.capture import capture_session
from rabbitcam_ml.dataset import (
    CorruptImageError,
    ImageRecord,
    RabbitCamDataset,
    discover_capture_sessions,
)
from rabbitcam_ml.transforms import build_transform


def _jpeg_bytes(tmp_path: Path, name: str = "source.jpg") -> bytes:
    path = tmp_path / name
    Image.new("RGB", (12, 8), (100, 20, 2)).save(path, format="JPEG")
    return path.read_bytes()


def test_capture_preserves_jpeg_and_writes_session_metadata(tmp_path: Path) -> None:
    jpeg = _jpeg_bytes(tmp_path)
    source = [
        JPEGFrame(jpeg, datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)),
        JPEGFrame(jpeg, datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)),
    ]
    result = capture_session(
        stream_url="http://camera:81/stream",
        label="lateral",
        lighting="ir",
        session_id="lateral-ir-001",
        output_dir=tmp_path / "captures",
        interval=0,
        max_frames=2,
        frame_source=source,
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.saved_frames == 2
    assert metadata["session_id"] == "lateral-ir-001"
    assert metadata["label"] == "lateral"
    assert metadata["lighting"] == "ir"
    assert metadata["stream_url"] == "http://camera:81/stream"
    assert metadata["status"] == "complete"
    assert all(
        (result.session_dir / item["filename"]).read_bytes() == jpeg
        for item in metadata["frames"]
    )


def test_discovery_reports_missing_and_corrupt_images(tmp_path: Path) -> None:
    session = tmp_path / "unknown-1"
    frames = session / "frames"
    frames.mkdir(parents=True)
    corrupt = frames / "corrupt.jpg"
    corrupt.write_bytes(b"not a jpeg")
    (session / "session.json").write_text(
        json.dumps(
            {
                "session_id": "unknown-1",
                "label": "unknown",
                "lighting": "visible",
                "frames": [
                    {"filename": "frames/corrupt.jpg", "timestamp": None},
                    {"filename": "frames/missing.jpg", "timestamp": None},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.warns(RuntimeWarning):
        discovery = discover_capture_sessions(tmp_path)
    assert discovery.records == ()
    assert {issue.kind for issue in discovery.issues} == {"corrupt", "missing"}


def test_dataset_returns_tensor_and_class_index(tmp_path: Path) -> None:
    image_path = tmp_path / "rabbit.jpg"
    Image.new("RGB", (20, 10), "white").save(image_path)
    dataset = RabbitCamDataset(
        [ImageRecord(image_path, "standing", "standing-visible-1", "visible")],
        transform=build_transform({"input_size": 24}),
    )
    tensor, target = dataset[0]
    assert tensor.shape == (3, 24, 24)
    assert target == 0


def test_dataset_runtime_corruption_has_context(tmp_path: Path) -> None:
    image_path = tmp_path / "bad.jpg"
    image_path.write_bytes(b"bad")
    dataset = RabbitCamDataset(
        [ImageRecord(image_path, "unknown", "unknown-1")],
        transform=build_transform(),
    )
    with pytest.raises(CorruptImageError, match="bad.jpg"):
        dataset[0]
