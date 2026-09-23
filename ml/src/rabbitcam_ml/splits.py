"""Deterministic, leakage-safe dataset splitting by capture session."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import warnings

from .labels import CLASS_NAMES, class_mapping, validate_label, validate_lighting


SPLIT_SCHEMA_VERSION = 1
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = ("train", "validation", "test")


class InsufficientSessionsError(ValueError):
    """Raised when independent sessions cannot support a meaningful split."""


@dataclass(frozen=True, slots=True)
class SplitSample:
    path: Path
    label: str
    session_id: str
    lighting: str
    captured_at: str | None
    split: str


@dataclass(frozen=True, slots=True)
class SplitManifest:
    samples: tuple[SplitSample, ...]
    seed: int
    ratios: dict[str, float]
    created_at: str
    source_root: str | None = None

    @property
    def session_splits(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for sample in self.samples:
            previous = result.setdefault(sample.session_id, sample.split)
            if previous != sample.split:
                raise ValueError(f"Session leakage detected for {sample.session_id!r}")
        return result

    def for_split(self, split: str) -> tuple[SplitSample, ...]:
        canonical = normalize_split_name(split)
        return tuple(sample for sample in self.samples if sample.split == canonical)

    def counts(self) -> dict[str, int]:
        counts = Counter(sample.split for sample in self.samples)
        return {name: counts[name] for name in SPLIT_NAMES}


def normalize_split_name(value: str) -> str:
    candidate = value.strip().lower()
    if candidate == "val":
        candidate = "validation"
    if candidate not in SPLIT_NAMES:
        raise ValueError(f"Invalid split {value!r}; expected train, validation/val, or test")
    return candidate


def _normalize_ratios(ratios: Mapping[str, float] | None) -> dict[str, float]:
    raw = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if "val" in raw:
        if "validation" in raw:
            raise ValueError("Use either 'val' or 'validation' in split ratios, not both")
        raw["validation"] = raw.pop("val")
    unknown = sorted(set(raw) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split ratio(s): {', '.join(unknown)}")
    result = {name: float(raw.get(name, 0.0)) for name in SPLIT_NAMES}
    if any(value < 0 for value in result.values()) or sum(result.values()) <= 0:
        raise ValueError("split ratios must be non-negative and have a positive sum")
    total = sum(result.values())
    return {name: value / total for name, value in result.items()}


def _session_order(session_ids: Sequence[str], *, seed: int, label: str) -> list[str]:
    def key(session_id: str) -> bytes:
        value = f"{seed}\0{label}\0{session_id}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return sorted(session_ids, key=key)


def _allocate_counts(count: int, ratios: Mapping[str, float], *, strict: bool) -> dict[str, int]:
    active = [name for name in SPLIT_NAMES if ratios[name] > 0]
    if count < len(active):
        message = (
            f"Need at least {len(active)} independent sessions per class to represent it in "
            f"all enabled splits, but found {count}"
        )
        if strict:
            raise InsufficientSessionsError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    raw = {name: count * ratios[name] for name in SPLIT_NAMES}
    allocated = {name: math.floor(raw[name]) for name in SPLIT_NAMES}
    for name in sorted(SPLIT_NAMES, key=lambda item: (raw[item] - allocated[item], ratios[item]), reverse=True):
        if sum(allocated.values()) >= count:
            break
        allocated[name] += 1

    if count >= len(active):
        for empty in (name for name in active if allocated[name] == 0):
            donors = [name for name in active if allocated[name] > 1]
            if not donors:
                raise InsufficientSessionsError("Unable to allocate at least one session per split")
            donor = max(donors, key=lambda name: (allocated[name], ratios[name]))
            allocated[donor] -= 1
            allocated[empty] += 1
    return allocated


def _field(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def create_split_manifest(
    records: Sequence[Any],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    strict: bool = True,
    output_path: str | Path | None = None,
    source_root: str | Path | None = None,
) -> SplitManifest:
    """Assign whole sessions to splits, stratified by session label.

    A session containing more than one label is rejected. When every class has
    at least three sessions (for the default ratios), every split contains that
    class. ``strict=False`` permits small exploratory datasets but emits a clear
    warning when this guarantee is impossible.
    """

    normalized_ratios = _normalize_ratios(ratios)
    records_by_session: dict[str, list[Any]] = defaultdict(list)
    label_by_session: dict[str, str] = {}
    for record in records:
        session_id = str(_field(record, "session_id", "")).strip()
        if not session_id:
            raise ValueError("Every record must have a non-empty session_id")
        label = validate_label(_field(record, "label"))
        previous = label_by_session.setdefault(session_id, label)
        if previous != label:
            raise ValueError(
                f"Session {session_id!r} contains multiple labels ({previous!r}, {label!r})"
            )
        records_by_session[session_id].append(record)

    sessions_by_label: dict[str, list[str]] = defaultdict(list)
    for session_id, label in label_by_session.items():
        sessions_by_label[label].append(session_id)

    missing_classes = [name for name in CLASS_NAMES if not sessions_by_label[name]]
    if missing_classes:
        message = f"No capture sessions found for class(es): {', '.join(missing_classes)}"
        if strict:
            raise InsufficientSessionsError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    session_splits: dict[str, str] = {}
    for label in CLASS_NAMES:
        ordered = _session_order(sessions_by_label[label], seed=seed, label=label)
        allocation = _allocate_counts(len(ordered), normalized_ratios, strict=strict)
        offset = 0
        for split_name in SPLIT_NAMES:
            next_offset = offset + allocation[split_name]
            for session_id in ordered[offset:next_offset]:
                session_splits[session_id] = split_name
            offset = next_offset

    samples: list[SplitSample] = []
    for session_id in sorted(records_by_session):
        split_name = session_splits[session_id]
        for record in sorted(records_by_session[session_id], key=lambda item: str(_field(item, "path"))):
            samples.append(
                SplitSample(
                    path=Path(_field(record, "path")).expanduser().resolve(),
                    label=validate_label(_field(record, "label")),
                    session_id=session_id,
                    lighting=validate_lighting(_field(record, "lighting", "unknown")),
                    captured_at=_field(record, "captured_at"),
                    split=split_name,
                )
            )

    manifest = SplitManifest(
        samples=tuple(samples),
        seed=int(seed),
        ratios=normalized_ratios,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_root=str(Path(source_root).expanduser().resolve()) if source_root else None,
    )
    # Accessing this property validates the central no-leakage invariant.
    manifest.session_splits
    if output_path is not None:
        save_split_manifest(manifest, output_path)
    return manifest


def save_split_manifest(manifest: SplitManifest, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    def serialized_path(sample_path: Path) -> str:
        try:
            return Path(os.path.relpath(sample_path, destination.parent)).as_posix()
        except ValueError:  # Different drives on Windows.
            return str(sample_path)

    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "created_at": manifest.created_at,
        "seed": manifest.seed,
        "ratios": manifest.ratios,
        "class_to_index": class_mapping(),
        "source_root": manifest.source_root,
        "session_splits": manifest.session_splits,
        "samples": [
            {
                "path": serialized_path(sample.path),
                "label": sample.label,
                "session_id": sample.session_id,
                "lighting": sample.lighting,
                "captured_at": sample.captured_at,
                "split": sample.split,
            }
            for sample in manifest.samples
        ],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def load_split_manifest(path: str | Path) -> SplitManifest:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read split manifest {source}: {exc}") from exc
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported split manifest schema {payload.get('schema_version')!r}; "
            f"expected {SPLIT_SCHEMA_VERSION}"
        )
    if payload.get("class_to_index") != class_mapping():
        raise ValueError("Split manifest class mapping is incompatible with this package")

    samples: list[SplitSample] = []
    try:
        for item in payload["samples"]:
            raw_path = Path(item["path"])
            resolved_path = raw_path if raw_path.is_absolute() else (source.parent / raw_path).resolve()
            samples.append(
                SplitSample(
                    path=resolved_path,
                    label=validate_label(item["label"]),
                    session_id=str(item["session_id"]),
                    lighting=validate_lighting(item.get("lighting", "unknown")),
                    captured_at=item.get("captured_at"),
                    split=normalize_split_name(item["split"]),
                )
            )
        manifest = SplitManifest(
            samples=tuple(samples),
            seed=int(payload["seed"]),
            ratios=_normalize_ratios(payload["ratios"]),
            created_at=str(payload["created_at"]),
            source_root=payload.get("source_root"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid split manifest {source}: {exc}") from exc

    stored_splits = payload.get("session_splits")
    if stored_splits is not None and stored_splits != manifest.session_splits:
        raise ValueError("Split manifest session assignments do not match its samples")
    return manifest


def generate_split_manifest(
    data_root: str | Path,
    output_path: str | Path,
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    strict: bool = True,
    verify_images: bool = True,
    detect_duplicates: bool = False,
) -> SplitManifest:
    """Discover captures, validate images, split sessions, and save JSON."""

    from .dataset import discover_capture_sessions

    discovery = discover_capture_sessions(
        data_root,
        verify_images=verify_images,
        strict=strict,
        detect_duplicates=detect_duplicates,
    )
    if not discovery.records:
        raise InsufficientSessionsError(f"No valid images found below {Path(data_root)}")
    return create_split_manifest(
        discovery.records,
        seed=seed,
        ratios=ratios,
        strict=strict,
        output_path=output_path,
        source_root=data_root,
    )
