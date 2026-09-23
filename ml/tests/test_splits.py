from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
from petcam_ml.dataset import ImageRecord
from petcam_ml.labels import CLASS_NAMES
from petcam_ml.splits import (
    InsufficientSessionsError,
    create_split_manifest,
    load_split_manifest,
)


def _records(root: Path, sessions_per_class: int = 5) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label in CLASS_NAMES:
        for session_number in range(sessions_per_class):
            session_id = f"{label}-{session_number}"
            for frame_number in range(2):
                records.append(
                    ImageRecord(
                        root / session_id / f"{frame_number}.jpg",
                        label,
                        session_id,
                        "ir" if session_number % 2 else "visible",
                    )
                )
    return records


def test_split_is_deterministic_and_never_leaks_sessions(tmp_path: Path) -> None:
    records = _records(tmp_path)
    first = create_split_manifest(records, seed=123)
    second = create_split_manifest(list(reversed(records)), seed=123)
    assert first.session_splits == second.session_splits

    observed: dict[str, set[str]] = defaultdict(set)
    for sample in first.samples:
        observed[sample.session_id].add(sample.split)
    assert all(len(splits) == 1 for splits in observed.values())

    for split in ("train", "validation", "test"):
        assert {sample.label for sample in first.for_split(split)} == set(CLASS_NAMES)


def test_saved_manifest_uses_relative_paths_and_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "manifests" / "split.json"
    created = create_split_manifest(_records(tmp_path), seed=7, output_path=output)
    loaded = load_split_manifest(output)
    assert loaded.session_splits == created.session_splits
    assert [sample.path for sample in loaded.samples] == [sample.path for sample in created.samples]
    assert '"class_to_index"' in output.read_text(encoding="utf-8")


def test_too_few_sessions_per_class_fails_in_strict_mode(tmp_path: Path) -> None:
    with pytest.raises(InsufficientSessionsError, match="at least 3"):
        create_split_manifest(_records(tmp_path, sessions_per_class=2))


def test_session_with_multiple_labels_is_rejected(tmp_path: Path) -> None:
    records = _records(tmp_path)
    records.append(ImageRecord(tmp_path / "bad.jpg", "lateral", "standing-0"))
    with pytest.raises(ValueError, match="multiple labels"):
        create_split_manifest(records)
