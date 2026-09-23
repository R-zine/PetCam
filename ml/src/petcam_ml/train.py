"""Headless, resumable training for the PetCam posture classifier.

The public :func:`train` function accepts either a loaded configuration mapping
or the path to a YAML configuration file.  Tests and downstream applications
may inject a model and dataloaders; normal CLI use builds both from the shared
model, dataset, and transform modules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .labels import CLASS_NAMES, class_mapping

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class ResumeInfo:
    """Position restored from ``last.pt``."""

    resumed: bool = False
    start_epoch: int = 0
    batches_to_skip: int = 0
    global_step: int = 0
    best_metric: float | None = None
    train_loss_sum: float = 0.0
    train_sample_count: int = 0
    loader_generator_state: Any | None = None


class _NullWriter:
    """SummaryWriter-compatible fallback for minimal installations."""

    def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def add_image(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _load_config(config: Mapping[str, Any] | str | os.PathLike[str]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    from .config import load_config

    return dict(load_config(config))


def _canonical_class_mapping() -> dict[str, int]:
    """Return the shared, serializable class mapping."""

    return class_mapping()


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available and otherwise to CPU."""

    if isinstance(requested, torch.device):
        device = requested
    else:
        value = str(requested).strip().lower()
        if value == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and torch and configure deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _resolve_path(value: str | os.PathLike[str], config_path: Path | None) -> Path:
    from .config import resolve_config_path

    return resolve_config_path(value, config_path)


def _build_dataloaders(
    config: Mapping[str, Any], config_path: Path | None
) -> dict[str, DataLoader[Any]]:
    from .dataset import PetCamDataset
    from .transforms import build_transform

    paths = _mapping(config.get("paths"))
    training = _mapping(config.get("training"))
    manifest_value = paths.get("split_manifest") or _mapping(config.get("data")).get(
        "split_manifest"
    )
    if not manifest_value:
        raise ValueError("Configuration must define paths.split_manifest")
    manifest_path = _resolve_path(manifest_value, config_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path}. Prepare a session-based split first."
        )

    preprocessing = _mapping(config.get("preprocessing"))
    train_transform_config = dict(preprocessing)
    train_transform_config["augmentation"] = _mapping(config.get("augmentation"))
    train_dataset = PetCamDataset.from_manifest(
        manifest_path,
        split="train",
        transform=build_transform(train_transform_config, training=True),
    )
    validation_dataset = PetCamDataset.from_manifest(
        manifest_path,
        split="validation",
        transform=build_transform(preprocessing, training=False),
    )
    if len(train_dataset) == 0:
        raise ValueError("The training split is empty")
    if len(validation_dataset) == 0:
        raise ValueError("The validation split is empty")

    batch_size = int(training.get("batch_size", 32))
    workers = int(training.get("workers", training.get("num_workers", 0)))
    if batch_size < 1 or workers < 0:
        raise ValueError("training.batch_size must be positive and workers cannot be negative")
    seed = int(training.get("seed", 1337))
    generator = torch.Generator().manual_seed(seed)
    common: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": resolve_device(training.get("device", "auto")).type == "cuda",
        "worker_init_fn": _worker_seed,
    }
    if workers:
        common["persistent_workers"] = True
    return {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        "validation": DataLoader(validation_dataset, shuffle=False, **common),
    }


def _build_model(
    config: Mapping[str, Any], model_factory: Callable[..., nn.Module] | None
) -> nn.Module:
    model_config = _mapping(config.get("model"))
    factory = model_factory
    if factory is None:
        from .models.factory import create_model

        factory = create_model
    return factory(
        model_name=str(model_config.get("name", "mobilenetv4_conv_small.e3600_r256_in1k")),
        num_classes=int(model_config.get("num_classes", len(CLASS_NAMES))),
        pretrained=bool(model_config.get("pretrained", True)),
    )


def _build_optimizer(model: nn.Module, training: Mapping[str, Any]) -> Optimizer:
    name = str(training.get("optimizer", "adamw")).lower()
    learning_rate = float(training.get("learning_rate", training.get("lr", 3e-4)))
    weight_decay = float(training.get("weight_decay", 1e-2))
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay cannot be negative")
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=float(training.get("momentum", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer {name!r}; expected adamw, adam, or sgd")


def _build_scheduler(optimizer: Optimizer, training: Mapping[str, Any], epochs: int) -> Any | None:
    scheduler_config = training.get("scheduler", "cosine")
    if isinstance(scheduler_config, Mapping):
        scheduler_name = str(scheduler_config.get("name", "cosine")).lower()
        scheduler_options = dict(scheduler_config)
    else:
        scheduler_name = str(scheduler_config or "none").lower()
        scheduler_options = {}
    if scheduler_name in {"none", "off", "false"}:
        return None
    if scheduler_name in {"cosine", "cosineannealing"}:
        minimum_lr = float(
            scheduler_options.get(
                "minimum_learning_rate", training.get("minimum_learning_rate", 1e-6)
            )
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=minimum_lr
        )
    if scheduler_name in {"step", "steplr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(
                scheduler_options.get("step_size", training.get("scheduler_step_size", 10))
            ),
            gamma=float(scheduler_options.get("gamma", training.get("scheduler_gamma", 0.1))),
        )
    if scheduler_name in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_options.get("factor", 0.2)),
            patience=int(scheduler_options.get("patience", 2)),
        )
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}")


def _make_grad_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(enabled=True)


def _targets_from_dataset(dataset: Any, class_to_idx: Mapping[str, int]) -> list[int]:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [int(item) for item in targets]
    records = getattr(dataset, "records", None)
    if records is not None:
        result: list[int] = []
        for record in records:
            label = getattr(record, "label", None)
            if label is None and isinstance(record, Mapping):
                label = record.get("label")
            if label in class_to_idx:
                result.append(class_to_idx[str(label)])
        return result
    return []


def _class_weight_tensor(
    setting: Any,
    train_loader: DataLoader[Any],
    class_to_idx: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor | None:
    if setting in (False, None, "none", "off"):
        return None
    if isinstance(setting, Sequence) and not isinstance(setting, (str, bytes)):
        weights = [float(value) for value in setting]
        if len(weights) != len(class_to_idx):
            raise ValueError("Explicit class_weights must contain one value per class")
        return torch.tensor(weights, dtype=torch.float32, device=device)
    targets = _targets_from_dataset(train_loader.dataset, class_to_idx)
    if not targets:
        print(
            "warning: class weighting requested but dataset does not expose labels; using no weights",
            file=sys.stderr,
        )
        return None
    counts = np.bincount(targets, minlength=len(class_to_idx)).astype(np.float64)
    missing = [name for name, index in class_to_idx.items() if counts[index] == 0]
    if missing:
        raise ValueError(f"Training split has no samples for classes: {', '.join(missing)}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _classifier_parameters(model: nn.Module) -> set[int]:
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            return {id(parameter) for parameter in classifier.parameters()}
    selected: set[int] = set()
    for name, parameter in model.named_parameters():
        parts = name.lower().split(".")
        if any(part in {"classifier", "head", "fc"} for part in parts):
            selected.add(id(parameter))
    if selected:
        return selected
    # A tiny injected Sequential model commonly has no named head. Its final
    # parameterized child is the least surprising classifier fallback.
    parameterized = [
        module
        for module in model.modules()
        if next(iter(module.parameters(recurse=False)), None) is not None
    ]
    if parameterized:
        return {id(parameter) for parameter in parameterized[-1].parameters(recurse=False)}
    raise ValueError("The model has no trainable classifier parameters")


def _set_warmup_state(model: nn.Module, head_only: bool) -> None:
    if not head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    classifier_ids = _classifier_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in classifier_ids)


def _extract_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        images = batch.get("image", batch.get("images"))
        targets = batch.get("label", batch.get("target", batch.get("labels")))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, targets = batch[0], batch[1]
    else:
        raise TypeError("A training batch must be (images, labels) or a mapping")
    if images is None or targets is None:
        raise ValueError("Training batch is missing images or labels")
    return torch.as_tensor(images), torch.as_tensor(targets, dtype=torch.long)


def _compute_metrics(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> dict[str, Any]:
    from .metrics import compute_classification_metrics

    try:
        return dict(compute_classification_metrics(y_true, y_pred, class_names=class_names))
    except TypeError:
        return dict(compute_classification_metrics(y_true, y_pred, labels=class_names))


def _validate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    class_names: list[str],
) -> tuple[float, dict[str, Any]]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            images, targets = _extract_batch(batch)
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            size = int(targets.numel())
            loss_sum += float(loss.detach().item()) * size
            sample_count += size
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    if sample_count == 0:
        raise ValueError("The validation loader produced no samples")
    return loss_sum / sample_count, _compute_metrics(y_true, y_pred, class_names)


def _metric_value(metrics: Mapping[str, Any], monitor: str) -> float:
    value: Any = metrics
    for part in monitor.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Checkpoint monitor {monitor!r} is not present in validation metrics")
        value = value[part]
    return float(value)


def _confusion_image(confusion_matrix: Any) -> torch.Tensor:
    matrix = torch.as_tensor(confusion_matrix, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.numel() == 0:
        return torch.zeros((3, 1, 1), dtype=torch.float32)
    maximum = float(matrix.max().item())
    normalized = matrix / maximum if maximum > 0 else matrix
    image = normalized.repeat_interleave(32, dim=0).repeat_interleave(32, dim=1)
    # A compact blue heatmap that has no matplotlib dependency.
    return torch.stack((0.15 * image, 0.45 * image, image), dim=0).clamp(0, 1)


def _flatten_epoch_row(
    epoch: int,
    global_step: int,
    train_loss: float,
    validation_loss: float,
    metrics: Mapping[str, Any],
    learning_rate: float,
    class_names: Sequence[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "lateral_recall": metrics.get("lateral_recall"),
        "learning_rate": learning_rate,
    }
    per_class = _mapping(metrics.get("per_class"))
    for name in class_names:
        values = _mapping(per_class.get(name))
        for metric_name in ("precision", "recall", "f1", "support"):
            row[f"{name}_{metric_name}"] = values.get(metric_name)
    return row


def _append_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_run_config(path: Path, config: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .config import save_config

        save_config(dict(config), path)
    except (ImportError, TypeError):
        # JSON is a valid YAML subset and keeps injected smoke tests lightweight.
        _atomic_json(path, dict(config))


def _snapshot_split_manifest(
    config: Mapping[str, Any],
    config_path: Path | None,
    run_dir: Path,
    *,
    fresh: bool,
) -> dict[str, Any] | None:
    """Pin the exact dataset split used by a run and return checkpoint metadata."""

    manifest_value = _mapping(config.get("paths")).get("split_manifest")
    if not manifest_value:
        # Dependency-injected unit tests may provide dataloaders without a file.
        return None
    source = _resolve_path(manifest_value, config_path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {source}. Prepare a session-based split first."
        )
    raw = source.read_bytes()
    try:
        payload = json.loads(raw)
        session_splits = payload["session_splits"]
        samples = payload["samples"]
        if not isinstance(session_splits, dict) or not isinstance(samples, list):
            raise TypeError("session_splits and samples have invalid types")
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid split manifest {source}: {exc}") from exc

    digest = hashlib.sha256(raw).hexdigest()
    snapshot = run_dir / "split-manifest.json"
    if snapshot.is_file() and not fresh:
        stored_digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if stored_digest != digest:
            raise RuntimeError(
                "The configured split manifest differs from the split snapshot already "
                f"used by this run ({snapshot}). Use the original manifest, choose a new "
                "paths.run_dir, or use --fresh intentionally."
            )
    else:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_name(f".{snapshot.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, snapshot)
        finally:
            temporary.unlink(missing_ok=True)

    return {
        "manifest_sha256": digest,
        "snapshot": snapshot.name,
        "seed": payload.get("seed"),
        "ratios": payload.get("ratios"),
        "session_splits": dict(session_splits),
        "sample_count": len(samples),
    }


def _make_writer(log_dir: Path, writer_factory: Callable[..., Any] | None) -> Any:
    if writer_factory is not None:
        return writer_factory(log_dir=str(log_dir))
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        print(
            "warning: tensorboard is not installed; training will continue without event logs",
            file=sys.stderr,
        )
        return _NullWriter()


def _log_epoch(
    writer: Any,
    epoch: int,
    global_step: int,
    train_loss: float,
    validation_loss: float,
    metrics: Mapping[str, Any],
    learning_rate: float,
) -> None:
    writer.add_scalar("loss/train", train_loss, global_step)
    writer.add_scalar("loss/validation", validation_loss, global_step)
    writer.add_scalar("validation/accuracy", float(metrics["accuracy"]), global_step)
    writer.add_scalar(
        "validation/balanced_accuracy", float(metrics["balanced_accuracy"]), global_step
    )
    writer.add_scalar("validation/macro_f1", float(metrics["macro_f1"]), global_step)
    writer.add_scalar("validation/lateral_recall", float(metrics["lateral_recall"]), global_step)
    writer.add_scalar("learning_rate", learning_rate, global_step)
    for name, values in _mapping(metrics.get("per_class")).items():
        for metric_name in ("precision", "recall", "f1"):
            if metric_name in values:
                writer.add_scalar(
                    f"validation/{name}_{metric_name}", float(values[metric_name]), global_step
                )
    writer.add_image(
        "validation/confusion_matrix",
        _confusion_image(metrics.get("confusion_matrix", [])),
        global_step=epoch,
    )
    writer.flush()


def _prune_periodic(checkpoint_dir: Path, keep: int) -> None:
    periodic = sorted(
        checkpoint_dir.glob("periodic-step-*.pt"),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
    )
    for old_checkpoint in periodic[: max(0, len(periodic) - max(0, keep))]:
        old_checkpoint.unlink(missing_ok=True)


def _fresh_run_cleanup(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.is_dir():
        for checkpoint in checkpoint_dir.glob("*.pt"):
            checkpoint.unlink(missing_ok=True)
    for artifact in (
        run_dir / "metrics.csv",
        run_dir / "summary.json",
        run_dir / "split-manifest.json",
    ):
        artifact.unlink(missing_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    if tensorboard_dir.is_dir():
        shutil.rmtree(tensorboard_dir)


def _checkpoint_payload(
    *,
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metric: float,
    class_to_idx: Mapping[str, int],
    model_name: str,
    preprocessing: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_summary: Mapping[str, Any] | None,
    epoch_complete: bool,
    batch_in_epoch: int,
    train_loss_sum: float,
    train_sample_count: int,
    loader_generator_state: Any | None,
    split_metadata: Mapping[str, Any] | None,
) -> None:
    from .models.checkpoint import save_checkpoint

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        global_step=global_step,
        best_metric=best_metric,
        class_to_idx=dict(class_to_idx),
        model_name=model_name,
        preprocessing_config=dict(preprocessing),
        training_config=dict(config),
        validation_summary=dict(validation_summary) if validation_summary is not None else None,
        metadata={
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch_complete": epoch_complete,
            "batch_in_epoch": batch_in_epoch,
            "train_loss_sum": train_loss_sum,
            "train_sample_count": train_sample_count,
            "loader_generator_state": loader_generator_state,
            "data_split": dict(split_metadata) if split_metadata is not None else None,
        },
    )


def _restore_if_available(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    scaler: Any,
    model_name: str,
    class_to_idx: Mapping[str, int],
    preprocessing: Mapping[str, Any],
    config: Mapping[str, Any],
    split_metadata: Mapping[str, Any] | None,
) -> ResumeInfo:
    if not checkpoint_path.is_file():
        return ResumeInfo()
    from .models.checkpoint import (
        load_checkpoint,
        restore_checkpoint,
        validate_checkpoint_compatibility,
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    stored_split = _mapping(_mapping(checkpoint.get("metadata")).get("data_split"))
    if split_metadata is not None and stored_split != dict(split_metadata):
        raise RuntimeError(
            f"Cannot auto-resume {checkpoint_path}: its dataset split metadata does not "
            "match the configured split. Use a new run directory or --fresh intentionally."
        )
    try:
        validate_checkpoint_compatibility(
            checkpoint,
            model_name=model_name,
            class_to_idx=dict(class_to_idx),
            preprocessing_config=dict(preprocessing),
            training_config=dict(config),
        )
    except (TypeError, ValueError) as exc:
        # Some versions intentionally compare only inference-critical metadata.
        if isinstance(exc, TypeError):
            validate_checkpoint_compatibility(
                checkpoint,
                model_name=model_name,
                class_to_idx=dict(class_to_idx),
                preprocessing_config=dict(preprocessing),
            )
        else:
            raise RuntimeError(
                f"Cannot auto-resume incompatible checkpoint {checkpoint_path}: {exc}. "
                "Use --fresh only if you intentionally want a new run."
            ) from exc
    state = restore_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_rng=True,
        strict=True,
    )
    metadata = _mapping(checkpoint.get("metadata"))
    epoch = int(getattr(state, "epoch", checkpoint.get("epoch", 0)))
    global_step = int(getattr(state, "global_step", checkpoint.get("global_step", 0)))
    best_metric_value = getattr(state, "best_metric", checkpoint.get("best_metric"))
    best_metric = float(best_metric_value) if best_metric_value is not None else None
    epoch_complete = bool(metadata.get("epoch_complete", True))
    return ResumeInfo(
        resumed=True,
        start_epoch=epoch + 1 if epoch_complete else epoch,
        batches_to_skip=0 if epoch_complete else int(metadata.get("batch_in_epoch", 0)),
        global_step=global_step,
        best_metric=best_metric,
        train_loss_sum=float(metadata.get("train_loss_sum", 0.0)),
        train_sample_count=int(metadata.get("train_sample_count", 0)),
        loader_generator_state=metadata.get("loader_generator_state"),
    )


def train(
    config: Mapping[str, Any] | str | os.PathLike[str],
    *,
    fresh: bool = False,
    model: nn.Module | None = None,
    model_factory: Callable[..., nn.Module] | None = None,
    dataloaders: Mapping[str, DataLoader[Any]] | None = None,
    writer_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Train or automatically resume a PetCam classifier.

    ``model``, ``model_factory``, ``dataloaders``, and ``writer_factory`` are
    dependency-injection hooks used by offline CPU smoke tests.  Production use
    only needs ``config`` and optionally ``fresh=True``.
    """

    config_path = None if isinstance(config, Mapping) else Path(config).resolve()
    loaded_config = _load_config(config)
    paths = _mapping(loaded_config.get("paths"))
    model_config = _mapping(loaded_config.get("model"))
    preprocessing = _mapping(loaded_config.get("preprocessing"))
    training = _mapping(loaded_config.get("training"))
    checkpoint_config = _mapping(loaded_config.get("checkpoint"))

    class_to_idx = _canonical_class_mapping()
    configured_classes = int(model_config.get("num_classes", len(class_to_idx)))
    if configured_classes != len(class_to_idx):
        raise ValueError(
            f"model.num_classes must be {len(class_to_idx)} for {list(class_to_idx)}, "
            f"not {configured_classes}"
        )
    epochs = int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1")
    seed = int(training.get("seed", 1337))
    seed_everything(seed, bool(training.get("deterministic", True)))
    device = resolve_device(training.get("device", "auto"))
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"

    run_dir = _resolve_path(paths.get("run_dir", "artifacts/runs/default"), config_path)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        _fresh_run_cleanup(run_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = _snapshot_split_manifest(loaded_config, config_path, run_dir, fresh=fresh)
    _save_run_config(run_dir / "config.yaml", loaded_config)

    loaders = (
        dict(dataloaders)
        if dataloaders is not None
        else _build_dataloaders(loaded_config, config_path)
    )
    train_loader = loaders.get("train")
    validation_loader = loaders.get("validation", loaders.get("val"))
    if train_loader is None or validation_loader is None:
        raise ValueError("dataloaders must contain train and validation entries")

    active_model = model if model is not None else _build_model(loaded_config, model_factory)
    active_model.to(device)
    optimizer = _build_optimizer(active_model, training)
    scheduler = _build_scheduler(optimizer, training, epochs)
    scaler = _make_grad_scaler(amp_enabled)
    model_name = str(model_config.get("name", active_model.__class__.__name__))
    resume = ResumeInfo()
    if not fresh:
        resume = _restore_if_available(
            checkpoint_path=checkpoint_dir / "last.pt",
            model=active_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_name=model_name,
            class_to_idx=class_to_idx,
            preprocessing=preprocessing,
            config=loaded_config,
            split_metadata=split_metadata,
        )
    if resume.resumed:
        print(
            f"Resuming {run_dir} at epoch {resume.start_epoch + 1}, "
            f"global step {resume.global_step}"
        )
    elif fresh:
        print(f"Starting fresh training run in {run_dir}")

    mode = str(checkpoint_config.get("mode", "max")).lower()
    if mode not in {"max", "min"}:
        raise ValueError("checkpoint.mode must be 'max' or 'min'")
    best_metric = resume.best_metric
    if best_metric is None or not math.isfinite(best_metric):
        best_metric = -math.inf if mode == "max" else math.inf
    monitor = str(checkpoint_config.get("monitor", "macro_f1"))
    checkpoint_every = int(checkpoint_config.get("checkpoint_every_steps", 0) or 0)
    keep_periodic = int(checkpoint_config.get("keep_periodic", 3))
    warmup_epochs = max(0, int(training.get("head_warmup_epochs", 0)))
    criterion = nn.CrossEntropyLoss(
        weight=_class_weight_tensor(
            training.get("class_weights", False), train_loader, class_to_idx, device
        )
    )
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    writer = _make_writer(run_dir / "tensorboard", writer_factory)
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "summary.json"
    global_step = resume.global_step
    latest_validation: dict[str, Any] | None = None
    current_epoch = max(0, resume.start_epoch)
    batch_in_epoch = resume.batches_to_skip
    train_loss_sum = resume.train_loss_sum
    train_sample_count = resume.train_sample_count
    loader_generator_state: Any | None = resume.loader_generator_state
    started_at = time.time()

    try:
        for epoch in range(resume.start_epoch, epochs):
            current_epoch = epoch
            skip_batches = resume.batches_to_skip if epoch == resume.start_epoch else 0
            batch_in_epoch = skip_batches
            _set_warmup_state(active_model, head_only=epoch < warmup_epochs)
            active_model.train()
            train_loss_sum = resume.train_loss_sum if skip_batches else 0.0
            train_sample_count = resume.train_sample_count if skip_batches else 0

            loader_generator = getattr(train_loader, "generator", None)
            if (
                skip_batches
                and loader_generator is not None
                and resume.loader_generator_state is not None
            ):
                loader_generator.set_state(resume.loader_generator_state)
            loader_generator_state = (
                loader_generator.get_state() if loader_generator is not None else None
            )
            iterator = iter(train_loader)
            if skip_batches:
                from .models.checkpoint import capture_rng_state, restore_rng_state

                checkpoint_rng_state = capture_rng_state()
                for skipped_index in range(skip_batches):
                    try:
                        next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            f"Checkpoint expects {skip_batches} completed batches, but the "
                            f"training loader ended after {skipped_index}; the dataset or batch "
                            "size changed"
                        ) from exc
                # Re-reading skipped augmented samples must not consume the RNG
                # state restored from the checkpoint.
                restore_rng_state(checkpoint_rng_state)

            for batch_index, batch in enumerate(iterator, start=skip_batches):
                images, targets = _extract_batch(batch)
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, amp_enabled):
                    logits = active_model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                size = int(targets.numel())
                train_loss_sum += float(loss.detach().item()) * size
                train_sample_count += size
                global_step += 1
                batch_in_epoch = batch_index + 1
                writer.add_scalar("loss/train_step", float(loss.detach().item()), global_step)

                if checkpoint_every > 0 and global_step % checkpoint_every == 0:
                    periodic = checkpoint_dir / f"periodic-step-{global_step:08d}.pt"
                    checkpoint_args = dict(
                        model=active_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        best_metric=best_metric,
                        class_to_idx=class_to_idx,
                        model_name=model_name,
                        preprocessing=preprocessing,
                        config=loaded_config,
                        validation_summary=latest_validation,
                        epoch_complete=False,
                        batch_in_epoch=batch_in_epoch,
                        train_loss_sum=train_loss_sum,
                        train_sample_count=train_sample_count,
                        loader_generator_state=loader_generator_state,
                        split_metadata=split_metadata,
                    )
                    _checkpoint_payload(path=periodic, **checkpoint_args)
                    _checkpoint_payload(path=checkpoint_dir / "last.pt", **checkpoint_args)
                    _prune_periodic(checkpoint_dir, keep_periodic)

            if train_sample_count == 0:
                raise ValueError("The training loader produced no samples")
            train_loss = train_loss_sum / train_sample_count
            validation_loss, validation_metrics = _validate(
                active_model,
                validation_loader,
                criterion,
                device,
                amp_enabled,
                class_names,
            )
            latest_validation = dict(validation_metrics)
            latest_validation["loss"] = validation_loss
            monitored_value = _metric_value(validation_metrics, monitor)
            improved = (
                monitored_value > best_metric if mode == "max" else monitored_value < best_metric
            )
            if improved:
                best_metric = monitored_value

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitored_value)
                else:
                    scheduler.step()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            row = _flatten_epoch_row(
                epoch + 1,
                global_step,
                train_loss,
                validation_loss,
                validation_metrics,
                learning_rate,
                class_names,
            )
            _append_metrics(metrics_path, row)
            _log_epoch(
                writer,
                epoch + 1,
                global_step,
                train_loss,
                validation_loss,
                validation_metrics,
                learning_rate,
            )
            checkpoint_args = dict(
                model=active_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                class_to_idx=class_to_idx,
                model_name=model_name,
                preprocessing=preprocessing,
                config=loaded_config,
                validation_summary=latest_validation,
                epoch_complete=True,
                batch_in_epoch=0,
                train_loss_sum=train_loss_sum,
                train_sample_count=train_sample_count,
                loader_generator_state=loader_generator_state,
                split_metadata=split_metadata,
            )
            _checkpoint_payload(path=checkpoint_dir / "last.pt", **checkpoint_args)
            if improved:
                _checkpoint_payload(path=checkpoint_dir / "best.pt", **checkpoint_args)

            summary = {
                "status": "completed" if epoch + 1 >= epochs else "running",
                "run_dir": str(run_dir),
                "device": str(device),
                "mixed_precision": amp_enabled,
                "resumed": resume.resumed,
                "epoch": epoch + 1,
                "epochs": epochs,
                "global_step": global_step,
                "best_metric": best_metric if math.isfinite(best_metric) else None,
                "monitor": monitor,
                "latest_validation": latest_validation,
                "elapsed_seconds": time.time() - started_at,
            }
            _atomic_json(summary_path, summary)
            print(
                f"epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
                f"macro_f1={float(validation_metrics['macro_f1']):.4f} "
                f"lateral_recall={float(validation_metrics['lateral_recall']):.4f} "
                f"lr={learning_rate:.3g}"
            )

        if resume.start_epoch >= epochs:
            summary = {
                "status": "already_completed",
                "run_dir": str(run_dir),
                "device": str(device),
                "mixed_precision": amp_enabled,
                "resumed": True,
                "epoch": resume.start_epoch,
                "epochs": epochs,
                "global_step": global_step,
                "best_metric": best_metric if math.isfinite(best_metric) else None,
                "monitor": monitor,
                "latest_validation": latest_validation,
                "elapsed_seconds": time.time() - started_at,
            }
            _atomic_json(summary_path, summary)
        return summary
    except KeyboardInterrupt:
        print("Training interrupted; saving resumable last.pt ...", file=sys.stderr)
        _checkpoint_payload(
            path=checkpoint_dir / "last.pt",
            model=active_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=current_epoch,
            global_step=global_step,
            best_metric=best_metric,
            class_to_idx=class_to_idx,
            model_name=model_name,
            preprocessing=preprocessing,
            config=loaded_config,
            validation_summary=latest_validation,
            epoch_complete=False,
            batch_in_epoch=batch_in_epoch,
            train_loss_sum=train_loss_sum,
            train_sample_count=train_sample_count,
            loader_generator_state=loader_generator_state,
            split_metadata=split_metadata,
        )
        _atomic_json(
            summary_path,
            {
                "status": "interrupted",
                "run_dir": str(run_dir),
                "epoch": current_epoch + 1,
                "epochs": epochs,
                "global_step": global_step,
                "best_metric": best_metric if math.isfinite(best_metric) else None,
                "monitor": monitor,
                "latest_validation": latest_validation,
                "elapsed_seconds": time.time() - started_at,
            },
        )
        raise
    finally:
        writer.flush()
        writer.close()


def train_from_config(
    config_path: str | os.PathLike[str], *, fresh: bool = False
) -> dict[str, Any]:
    """CLI-friendly wrapper around :func:`train`."""

    return train(config_path, fresh=fresh)


run_training = train


__all__ = [
    "ResumeInfo",
    "resolve_device",
    "run_training",
    "seed_everything",
    "train",
    "train_from_config",
]
