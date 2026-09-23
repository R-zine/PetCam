"""Rich, atomic and resumable PyTorch checkpoints."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

CHECKPOINT_VERSION = 1


class CheckpointError(RuntimeError):
    """Base exception for invalid or unreadable RabbitCam checkpoints."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised before loading state from a materially different run."""


@dataclass(frozen=True)
class ResumeState:
    """Training cursor restored from a checkpoint."""

    epoch: int
    global_step: int
    best_metric: float
    validation_summary: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU and (when available) CUDA RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any] | None) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""

    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        raise CheckpointError("Checkpoint RNG state is invalid") from exc


def _state_dict(component: Any | None) -> Mapping[str, Any] | None:
    return component.state_dict() if component is not None else None


def build_checkpoint(
    *,
    model: nn.Module,
    epoch: int,
    global_step: int,
    best_metric: float,
    class_to_idx: Mapping[str, int],
    model_name: str,
    preprocessing_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    validation_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical serializable checkpoint payload."""

    mapping = {str(label): int(index) for label, index in class_to_idx.items()}
    _validate_class_mapping(mapping)
    if not model_name:
        raise ValueError("model_name must be non-empty")
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_name": str(model_name),
        "class_to_idx": mapping,
        "preprocessing_config": copy.deepcopy(dict(preprocessing_config)),
        "training_config": copy.deepcopy(dict(training_config)),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _state_dict(optimizer),
        "scheduler_state_dict": _state_dict(scheduler),
        "scaler_state_dict": _state_dict(scaler),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "validation_summary": (
            copy.deepcopy(dict(validation_summary)) if validation_summary is not None else None
        ),
        "metadata": copy.deepcopy(dict(metadata or {})),
    }


def save_checkpoint(
    path: str | os.PathLike[str],
    checkpoint: Mapping[str, Any] | None = None,
    *,
    model: nn.Module | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float = 0.0,
    class_to_idx: Mapping[str, int] | None = None,
    model_name: str | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
    training_config: Mapping[str, Any] | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    validation_summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a checkpoint and return its resolved destination.

    Callers may pass a complete ``checkpoint`` mapping or the individual
    training components.  The temporary file is created beside the target so
    that ``os.replace`` remains atomic on the same filesystem.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        if model is not None:
            raise ValueError("Pass a checkpoint payload or model components, not both")
        payload = dict(checkpoint)
        payload.setdefault("checkpoint_version", CHECKPOINT_VERSION)
        payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
    else:
        if model is None or class_to_idx is None or model_name is None:
            raise ValueError("model, class_to_idx, and model_name are required")
        payload = build_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            class_to_idx=class_to_idx,
            model_name=model_name,
            preprocessing_config=preprocessing_config or {},
            training_config=training_config or {},
            validation_summary=validation_summary,
            metadata=metadata,
            rng_state=rng_state,
        )

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise CheckpointError(f"Could not save checkpoint to {destination}") from exc
    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a RabbitCam checkpoint."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {source}")
    try:
        # Explicit weights_only=False is necessary because a rich resumable
        # checkpoint includes Python and NumPy RNG state in addition to tensors.
        try:
            value = torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:  # PyTorch versions predating the weights_only argument
            value = torch.load(source, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(f"Could not load checkpoint {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Checkpoint {source} is not a mapping")
    required = {"model_state_dict", "model_name", "class_to_idx"}
    missing = sorted(required.difference(value))
    if missing:
        raise CheckpointError(f"Checkpoint {source} is missing: {', '.join(missing)}")
    _validate_class_mapping(value["class_to_idx"])
    return value


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
    strict: bool = True,
) -> ResumeState:
    """Restore model and optional training components from a loaded payload."""

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        for component, key in (
            (optimizer, "optimizer_state_dict"),
            (scheduler, "scheduler_state_dict"),
            (scaler, "scaler_state_dict"),
        ):
            state = checkpoint.get(key)
            if component is not None and state is not None:
                component.load_state_dict(state)
        if restore_rng:
            restore_rng_state(checkpoint.get("rng_state"))
    except CheckpointError:
        raise
    except Exception as exc:
        raise CheckpointError(f"Could not restore checkpoint state: {exc}") from exc
    return ResumeState(
        epoch=int(checkpoint.get("epoch", -1)),
        global_step=int(checkpoint.get("global_step", 0)),
        best_metric=float(checkpoint.get("best_metric", 0.0)),
        validation_summary=checkpoint.get("validation_summary"),
        metadata=checkpoint.get("metadata"),
    )


def _validate_class_mapping(class_to_idx: Mapping[str, int]) -> None:
    if not isinstance(class_to_idx, Mapping) or not class_to_idx:
        raise CheckpointError("class_to_idx must be a non-empty mapping")
    try:
        indices = [int(index) for index in class_to_idx.values()]
    except (TypeError, ValueError) as exc:
        raise CheckpointError("class_to_idx values must be integers") from exc
    expected = list(range(len(indices)))
    if sorted(indices) != expected or len(set(class_to_idx)) != len(indices):
        raise CheckpointError(f"class_to_idx indices must be contiguous {expected}")


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _material_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop runtime-only values that are safe to change while resuming."""

    result = copy.deepcopy(dict(config))
    result.pop("checkpoint", None)
    paths = result.get("paths")
    if isinstance(paths, dict):
        for key in ("run_dir", "artifacts_dir", "tensorboard_dir"):
            paths.pop(key, None)
    training = result.get("training")
    if isinstance(training, dict):
        for key in ("epochs", "device", "workers", "num_workers"):
            training.pop(key, None)
    return _canonical(result)


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    class_to_idx: Mapping[str, int],
    preprocessing_config: Mapping[str, Any],
    training_config: Mapping[str, Any] | None = None,
) -> None:
    """Raise a clear error when automatic resume would be unsafe."""

    differences: list[str] = []
    if str(checkpoint.get("model_name")) != str(model_name):
        differences.append(
            f"model_name ({checkpoint.get('model_name')!r} != {model_name!r})"
        )
    if _canonical(checkpoint.get("class_to_idx", {})) != _canonical(class_to_idx):
        differences.append("class mapping")
    stored_preprocessing = checkpoint.get(
        "preprocessing_config", checkpoint.get("preprocessing", {})
    )
    if _canonical(stored_preprocessing) != _canonical(preprocessing_config):
        differences.append("preprocessing configuration")
    if training_config is not None:
        stored_config = checkpoint.get("training_config")
        if not isinstance(stored_config, Mapping):
            differences.append("missing training configuration")
        elif _material_training_config(stored_config) != _material_training_config(training_config):
            differences.append("material training configuration")
    if differences:
        joined = ", ".join(differences)
        raise IncompatibleCheckpointError(
            f"Checkpoint is incompatible with this run: {joined}. "
            "Use --fresh only if starting a new run is intentional."
        )


def inspect_checkpoint(
    checkpoint_or_path: Mapping[str, Any] | str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a JSON-safe summary without large tensor/optimizer state."""

    checkpoint = (
        load_checkpoint(checkpoint_or_path)
        if isinstance(checkpoint_or_path, (str, os.PathLike))
        else dict(checkpoint_or_path)
    )
    state = checkpoint.get("model_state_dict", {})
    parameter_tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    parameter_count = int(sum(value.numel() for value in parameter_tensors))
    metadata = checkpoint.get("metadata") or {}
    summary = {
        "checkpoint_version": int(checkpoint.get("checkpoint_version", 0)),
        "saved_at": checkpoint.get("saved_at"),
        "model_name": checkpoint.get("model_name"),
        "model_kwargs": metadata.get("model_kwargs", {}),
        "parameter_count": parameter_count,
        "class_to_idx": checkpoint.get("class_to_idx"),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_metric": float(checkpoint.get("best_metric", 0.0)),
        "preprocessing_config": checkpoint.get(
            "preprocessing_config", checkpoint.get("preprocessing", {})
        ),
        "training_config": checkpoint.get("training_config", {}),
        "validation_summary": checkpoint.get("validation_summary"),
        "metadata": metadata,
        "has_optimizer_state": checkpoint.get("optimizer_state_dict") is not None,
        "has_scheduler_state": checkpoint.get("scheduler_state_dict") is not None,
        "has_scaler_state": checkpoint.get("scaler_state_dict") is not None,
        "has_rng_state": checkpoint.get("rng_state") is not None,
    }
    # Round-tripping catches accidental non-JSON values in user metadata while
    # preserving the original rich checkpoint on disk.
    try:
        return json.loads(json.dumps(_canonical(summary)))
    except (TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint metadata is not JSON serializable") from exc


__all__ = [
    "CHECKPOINT_VERSION",
    "CheckpointError",
    "IncompatibleCheckpointError",
    "ResumeState",
    "build_checkpoint",
    "capture_rng_state",
    "inspect_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "validate_checkpoint_compatibility",
]
