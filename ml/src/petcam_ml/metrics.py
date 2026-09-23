"""Deterministic, JSON-safe classification metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_CLASS_NAMES = ("standing", "lateral", "unknown")


def _as_flat_list(values: Any) -> list[Any]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"Expected one-dimensional labels, got shape {array.shape}")
    return array.tolist()


def _to_indices(values: Any, class_names: Sequence[str]) -> list[int]:
    label_to_idx = {name: index for index, name in enumerate(class_names)}
    result: list[int] = []
    for value in _as_flat_list(values):
        if isinstance(value, str):
            if value not in label_to_idx:
                raise ValueError(f"Unknown class label {value!r}")
            result.append(label_to_idx[value])
        else:
            index = int(value)
            if index < 0 or index >= len(class_names):
                raise ValueError(f"Class index {index} is outside [0, {len(class_names)})")
            result.append(index)
    return result


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> list[list[int]]:
    """Return a true-label-by-predicted-label confusion matrix."""

    names = tuple(class_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("class_names must be non-empty and unique")
    truth = _to_indices(y_true, names)
    predicted = _to_indices(y_pred, names)
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same number of items")
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        matrix[expected, actual] += 1
    return matrix.tolist()


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the complete metric set used by training and evaluation.

    ``labels`` is accepted as an alias for ``class_names``.  All values are
    native Python objects and can be written directly with ``json.dump``.
    Balanced accuracy averages recall only over classes represented in the
    ground truth; macro-F1 includes every configured class.
    """

    if labels is not None:
        if tuple(class_names) != DEFAULT_CLASS_NAMES and tuple(class_names) != tuple(labels):
            raise ValueError("Pass class_names or labels, not conflicting values for both")
        class_names = labels
    names = tuple(str(name) for name in class_names)
    matrix = np.asarray(compute_confusion_matrix(y_true, y_pred, names), dtype=np.int64)
    total = int(matrix.sum())

    per_class: dict[str, dict[str, float | int]] = {}
    recalls_present: list[float] = []
    f1_values: list[float] = []
    counts: dict[str, int] = {}
    for index, name in enumerate(names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            recalls_present.append(recall)
        f1_values.append(f1)
        counts[name] = support
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
            "predicted": predicted_count,
        }

    accuracy = float(np.trace(matrix) / total) if total else 0.0
    balanced_accuracy = float(np.mean(recalls_present)) if recalls_present else 0.0
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    lateral_recall = float(per_class.get("lateral", {}).get("recall", 0.0))
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "lateral_recall": lateral_recall,
        "class_names": list(names),
        "counts": counts,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def metrics_from_logits(
    targets: Any,
    logits: Any,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> dict[str, Any]:
    """Compute metrics after applying argmax to an ``N x C`` logits array."""

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    array = np.asarray(logits)
    if array.ndim != 2 or array.shape[1] != len(class_names):
        raise ValueError(f"Expected logits shaped (N, {len(class_names)}), got {array.shape}")
    return compute_classification_metrics(targets, np.argmax(array, axis=1), class_names)


classification_metrics = compute_classification_metrics

__all__ = [
    "DEFAULT_CLASS_NAMES",
    "classification_metrics",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "metrics_from_logits",
]
