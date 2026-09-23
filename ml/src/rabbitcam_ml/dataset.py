"""Capture-session discovery and PyTorch dataset support."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal
import warnings

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset

from .labels import CLASS_TO_INDEX, validate_label, validate_lighting


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    label: str
    session_id: str
    lighting: str = "unknown"
    captured_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "label", validate_label(self.label))
        object.__setattr__(self, "lighting", validate_lighting(self.lighting))
        if not self.session_id:
            raise ValueError("session_id cannot be empty")


IssueKind = Literal["invalid_metadata", "missing", "corrupt", "duplicate"]


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    kind: IssueKind
    path: Path
    message: str
    duplicate_of: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetDiscovery:
    records: tuple[ImageRecord, ...]
    issues: tuple[DatasetIssue, ...]
    session_count: int


class DatasetValidationError(ValueError):
    """Raised when capture metadata or an image is invalid in strict mode."""


class CorruptImageError(RuntimeError):
    """Raised when an image becomes unreadable while loading a dataset."""


def _readable_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return str(exc)


def discover_capture_sessions(
    root: str | Path,
    *,
    verify_images: bool = True,
    strict: bool = False,
    detect_duplicates: bool = False,
) -> DatasetDiscovery:
    """Discover records from all ``session.json`` files below ``root``.

    Missing and corrupt files are reported as structured issues and omitted.
    With ``strict=True``, the first issue raises ``DatasetValidationError``.
    Exact duplicate detection is optional and duplicate records are reported
    but retained; callers may decide whether to remove them.
    """

    data_root = Path(root).expanduser().resolve()
    metadata_paths = sorted(data_root.rglob("session.json"))
    records: list[ImageRecord] = []
    issues: list[DatasetIssue] = []
    digests: dict[str, Path] = {}

    def report(issue: DatasetIssue) -> None:
        if strict:
            raise DatasetValidationError(issue.message)
        issues.append(issue)
        warnings.warn(issue.message, RuntimeWarning, stacklevel=2)

    for metadata_path in metadata_paths:
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            session_id = str(metadata["session_id"])
            label = validate_label(metadata["label"])
            lighting = validate_lighting(metadata.get("lighting", "unknown"))
            frames = metadata["frames"]
            if not isinstance(frames, list):
                raise TypeError("frames must be a list")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            report(
                DatasetIssue(
                    "invalid_metadata",
                    metadata_path,
                    f"Invalid session metadata {metadata_path}: {exc}",
                )
            )
            continue

        for frame in frames:
            try:
                relative_name = frame["filename"]
                if not isinstance(relative_name, str):
                    raise TypeError("filename must be a string")
                image_path = (metadata_path.parent / relative_name).resolve()
                # A manifest must not escape its own session directory.
                image_path.relative_to(metadata_path.parent.resolve())
                captured_at = frame.get("timestamp")
            except (KeyError, TypeError, ValueError) as exc:
                report(
                    DatasetIssue(
                        "invalid_metadata",
                        metadata_path,
                        f"Invalid frame entry in {metadata_path}: {exc}",
                    )
                )
                continue
            if not image_path.is_file():
                report(
                    DatasetIssue("missing", image_path, f"Missing image referenced by metadata: {image_path}")
                )
                continue
            if verify_images and (error := _readable_image(image_path)) is not None:
                report(DatasetIssue("corrupt", image_path, f"Corrupt image {image_path}: {error}"))
                continue
            if detect_duplicates:
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if digest in digests:
                    report(
                        DatasetIssue(
                            "duplicate",
                            image_path,
                            f"Exact duplicate image {image_path} matches {digests[digest]}",
                            duplicate_of=digests[digest],
                        )
                    )
                else:
                    digests[digest] = image_path
            records.append(ImageRecord(image_path, label, session_id, lighting, captured_at))

    return DatasetDiscovery(tuple(records), tuple(issues), len(metadata_paths))


def discover_samples(root: str | Path, **kwargs: Any) -> list[ImageRecord]:
    """Convenience wrapper returning only valid discovered records."""

    return list(discover_capture_sessions(root, **kwargs).records)


def load_records_from_split_manifest(
    manifest_path: str | Path, split: str
) -> list[ImageRecord]:
    """Load one split as dataset records without duplicating manifest logic."""

    from .splits import load_split_manifest

    samples = load_split_manifest(manifest_path).for_split(split)
    return [
        ImageRecord(
            path=sample.path,
            label=sample.label,
            session_id=sample.session_id,
            lighting=sample.lighting,
            captured_at=sample.captured_at,
        )
        for sample in samples
    ]


class RabbitCamDataset(Dataset[tuple[torch.Tensor, int]]):
    """Simple image classification dataset backed by explicit records."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.records = tuple(records)
        self.transform = transform

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> "RabbitCamDataset":
        return cls(load_records_from_split_manifest(manifest_path, split), transform=transform)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        try:
            with Image.open(record.path) as source:
                source.load()
                image = source.copy()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise CorruptImageError(f"Unable to load image {record.path}: {exc}") from exc
        if self.transform is None:
            raise RuntimeError("RabbitCamDataset requires a transform that returns a tensor")
        return self.transform(image), CLASS_TO_INDEX[record.label]
