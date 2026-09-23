"""Capture labeled JPEG frames from a RabbitCam MJPEG stream."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any

from .camera.mjpeg import JPEGFrame, MJPEGStream
from .labels import validate_label, validate_lighting


LOGGER = logging.getLogger(__name__)
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SESSION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CaptureResult:
    session_dir: Path
    metadata_path: Path
    saved_frames: int
    interrupted: bool


def validate_session_id(value: str) -> str:
    """Validate a session ID before it is used as a directory name."""

    session_id = value.strip()
    if not SESSION_ID_PATTERN.fullmatch(session_id) or session_id in {".", ".."}:
        raise ValueError(
            "session ID must start with an alphanumeric character and contain "
            "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
        )
    return session_id


def _utc_iso(value: datetime | None = None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def capture_session(
    *,
    stream_url: str,
    label: str,
    lighting: str,
    session_id: str,
    output_dir: str | Path,
    interval: float = 1.0,
    duration: float | None = None,
    timeout: float = 10.0,
    max_frames: int | None = None,
    frame_source: Iterable[JPEGFrame] | None = None,
) -> CaptureResult:
    """Capture one labeled session while preserving original JPEG bytes.

    The session is written to ``output_dir/session_id``. Existing non-empty
    session directories are refused to prevent accidental mixing of separate
    capture sessions. ``frame_source`` exists for tests and offline adapters.
    """

    canonical_label = validate_label(label)
    canonical_lighting = validate_lighting(lighting)
    canonical_session_id = validate_session_id(session_id)
    if interval < 0:
        raise ValueError("interval must be non-negative")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    session_dir = Path(output_dir).expanduser().resolve() / canonical_session_id
    frames_dir = session_dir / "frames"
    metadata_path = session_dir / "session.json"
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(
            f"Capture session directory is not empty: {session_dir}. "
            "Use a new session ID so correlated sessions remain distinct."
        )
    frames_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    metadata: dict[str, Any] = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "session_id": canonical_session_id,
        "label": canonical_label,
        "lighting": canonical_lighting,
        "stream_url": stream_url,
        "started_at": _utc_iso(started_at),
        "ended_at": None,
        "status": "capturing",
        "capture_interval_seconds": interval,
        "frames": [],
    }
    _write_json_atomic(metadata_path, metadata)

    started_monotonic = time.monotonic()
    deadline = started_monotonic + duration if duration is not None else None
    next_save_time = started_monotonic
    interrupted = False
    saved_frames = 0

    def should_stop() -> bool:
        return deadline is not None and time.monotonic() >= deadline

    source = frame_source
    if source is None:
        source = MJPEGStream(stream_url, timeout=timeout).iter_frames(
            reconnect=True, stop_requested=should_stop
        )

    LOGGER.info(
        "Capturing session %s (%s, %s) from %s into %s",
        canonical_session_id,
        canonical_label,
        canonical_lighting,
        stream_url,
        session_dir,
    )
    try:
        for frame in source:
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                break
            if now < next_save_time:
                continue

            saved_frames += 1
            captured_at = frame.captured_at
            stamp = captured_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            filename = f"{stamp}_{saved_frames:06d}.jpg"
            relative_path = Path("frames") / filename
            destination = session_dir / relative_path
            destination.write_bytes(frame.data)
            metadata["frames"].append(
                {
                    "filename": relative_path.as_posix(),
                    "timestamp": _utc_iso(captured_at),
                    "size_bytes": len(frame.data),
                }
            )
            LOGGER.info("Saved frame %d: %s", saved_frames, destination)
            next_save_time = now + interval
            if max_frames is not None and saved_frames >= max_frames:
                break
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.info("Capture interrupted; finalizing metadata")
    finally:
        metadata["ended_at"] = _utc_iso()
        metadata["status"] = "interrupted" if interrupted else "complete"
        metadata["frame_count"] = saved_frames
        _write_json_atomic(metadata_path, metadata)

    LOGGER.info("Capture complete: %d frame(s) in %s", saved_frames, session_dir)
    return CaptureResult(
        session_dir=session_dir,
        metadata_path=metadata_path,
        saved_frames=saved_frames,
        interrupted=interrupted,
    )
