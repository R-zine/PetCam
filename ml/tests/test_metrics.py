from __future__ import annotations

import pytest
from petcam_ml.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    metrics_from_logits,
)


def test_metrics_include_per_class_counts_and_lateral_recall() -> None:
    truth = [0, 0, 1, 1, 2, 2]
    predicted = [0, 1, 1, 1, 0, 2]

    result = compute_classification_metrics(truth, predicted)

    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert result["lateral_recall"] == pytest.approx(1.0)
    assert result["counts"] == {"standing": 2, "lateral": 2, "unknown": 2}
    assert result["per_class"]["lateral"]["support"] == 2


def test_metrics_accept_string_labels_and_logits() -> None:
    matrix = compute_confusion_matrix(
        ["standing", "lateral", "unknown"],
        ["standing", "unknown", "unknown"],
    )
    assert matrix == [[1, 0, 0], [0, 0, 1], [0, 0, 1]]

    result = metrics_from_logits(
        [0, 1, 2],
        [[4.0, 0.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 2.0]],
    )
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        compute_classification_metrics([0], [0, 1])
    with pytest.raises(ValueError, match="Unknown class"):
        compute_classification_metrics(["rabbit"], ["standing"])
    with pytest.raises(ValueError, match="logits"):
        metrics_from_logits([0], [[1.0, 2.0]])
