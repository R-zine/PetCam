"""Checkpoint evaluation helpers used by the CLI and tests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from .inference import Predictor, resolve_device
from .metrics import compute_classification_metrics


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, Any]:
    if isinstance(batch, Mapping):
        images = batch.get("image", batch.get("images"))
        targets = batch.get("target", batch.get("label", batch.get("labels")))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, targets = batch[0], batch[1]
    else:
        raise ValueError("Evaluation batches must contain images and labels")
    if not isinstance(images, torch.Tensor):
        raise TypeError("Evaluation images must be tensors")
    if targets is None:
        raise ValueError("Evaluation batch has no labels")
    return images, targets


def _targets_as_indices(targets: Any, class_to_idx: Mapping[str, int]) -> list[int]:
    if isinstance(targets, torch.Tensor):
        return [int(value) for value in targets.detach().cpu().reshape(-1).tolist()]
    if isinstance(targets, str):
        targets = [targets]
    result = []
    for target in targets:
        if isinstance(target, str):
            if target not in class_to_idx:
                raise ValueError(f"Unknown evaluation label {target!r}")
            result.append(int(class_to_idx[target]))
        else:
            result.append(int(target))
    return result


def evaluate_model(
    model: nn.Module,
    dataloader: Iterable[Any],
    *,
    class_to_idx: Mapping[str, int],
    device: str | torch.device = "auto",
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate an already constructed model without altering its weights."""

    resolved_device = resolve_device(device)
    ordered_names = [
        name for name, _ in sorted(class_to_idx.items(), key=lambda item: int(item[1]))
    ]
    model.to(resolved_device)
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    loss_total = 0.0
    loss_items = 0

    with torch.inference_mode():
        for batch in dataloader:
            images, targets = _unpack_batch(batch)
            target_indices = _targets_as_indices(targets, class_to_idx)
            target_tensor = torch.tensor(target_indices, dtype=torch.long, device=resolved_device)
            logits = model(images.to(resolved_device))
            if isinstance(logits, Mapping):
                logits = logits.get("logits")
            elif isinstance(logits, (tuple, list)):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                raise RuntimeError("Model did not return a two-dimensional logits tensor")
            if logits.shape[0] != len(target_indices) or logits.shape[1] != len(ordered_names):
                raise RuntimeError(
                    f"Unexpected logits shape {tuple(logits.shape)} for "
                    f"{len(target_indices)} samples and {len(ordered_names)} classes"
                )
            if criterion is not None:
                batch_loss = criterion(logits, target_tensor)
                loss_total += float(batch_loss.detach().item()) * len(target_indices)
                loss_items += len(target_indices)
            truth.extend(target_indices)
            predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())

    result = compute_classification_metrics(truth, predicted, ordered_names)
    result["loss"] = loss_total / loss_items if loss_items else None
    return result


def save_evaluation_results(results: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write evaluation JSON."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(results, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return destination


def save_confusion_matrix_image(
    matrix: Sequence[Sequence[int]],
    class_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Save a dependency-light labeled confusion matrix PNG."""

    count = len(class_names)
    if count == 0 or len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError("Confusion matrix shape must match class_names")
    cell = 104
    margin_left = 120
    margin_top = 70
    image = Image.new("RGB", (margin_left + cell * count, margin_top + cell * count), "white")
    draw = ImageDraw.Draw(image)
    maximum = max((int(value) for row in matrix for value in row), default=1) or 1
    for row in range(count):
        for column in range(count):
            value = int(matrix[row][column])
            intensity = int(245 - 175 * value / maximum)
            fill = (intensity, intensity, 255)
            x0 = margin_left + column * cell
            y0 = margin_top + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="black")
            draw.text((x0 + cell // 2 - 8, y0 + cell // 2 - 6), str(value), fill="black")
    for index, name in enumerate(class_names):
        draw.text((margin_left + index * cell + 8, 30), str(name), fill="black")
        draw.text((8, margin_top + index * cell + cell // 2 - 6), str(name), fill="black")
    draw.text((margin_left, 8), "Predicted label", fill="black")
    draw.text((8, margin_top - 20), "True label", fill="black")
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    output_json: str | Path | None = None,
    confusion_matrix_image: str | Path | None = None,
) -> dict[str, Any]:
    """Load any PetCam checkpoint and evaluate one manifest split."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    predictor = Predictor(checkpoint_path, device=device)
    from .dataset import PetCamDataset

    dataset = PetCamDataset.from_manifest(
        manifest_path,
        split,
        predictor.transform,
    )
    if len(dataset) == 0:
        raise ValueError(f"Manifest split {split!r} contains no usable samples")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metrics = evaluate_model(
        predictor.model,
        dataloader,
        class_to_idx=predictor.class_to_idx,
        device=predictor.device,
        criterion=nn.CrossEntropyLoss(),
    )
    result: dict[str, Any] = {
        "checkpoint": str(predictor.checkpoint_path),
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "split": split,
        **metrics,
    }
    default_output = predictor.checkpoint_path.with_name(
        f"{predictor.checkpoint_path.stem}-{split}-evaluation.json"
    )
    json_path = save_evaluation_results(result, output_json or default_output)
    result["output_json"] = str(json_path)
    if confusion_matrix_image is not None:
        matrix_path = save_confusion_matrix_image(
            result["confusion_matrix"], result["class_names"], confusion_matrix_image
        )
        result["confusion_matrix_image"] = str(matrix_path)
    return result


__all__ = [
    "evaluate_checkpoint",
    "evaluate_model",
    "save_confusion_matrix_image",
    "save_evaluation_results",
]
