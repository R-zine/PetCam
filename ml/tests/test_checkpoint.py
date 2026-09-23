from __future__ import annotations

from pathlib import Path

import pytest
import torch
from petcam_ml.models.checkpoint import (
    IncompatibleCheckpointError,
    inspect_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from petcam_ml.models.factory import create_model

CLASS_TO_IDX = {"standing": 0, "lateral": 1, "unknown": 2}
PREPROCESSING = {"image_size": 32, "grayscale": True}


def test_checkpoint_round_trip_restores_full_training_state(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = create_model("tiny_cnn", pretrained=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    sample = torch.rand(2, 3, 16, 16)
    model(sample).sum().backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "nested" / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        global_step=87,
        best_metric=0.71,
        class_to_idx=CLASS_TO_IDX,
        model_name="tiny_cnn",
        preprocessing_config=PREPROCESSING,
        training_config={"training": {"lr": 0.01, "epochs": 5}},
        validation_summary={"macro_f1": 0.71},
    )

    assert path.is_file()
    assert list(path.parent.glob("*.tmp")) == []
    checkpoint = load_checkpoint(path)
    restored_model = create_model("tiny_cnn", pretrained=False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01)
    state = restore_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        restore_rng=False,
    )

    assert state.epoch == 4
    assert state.global_step == 87
    assert state.best_metric == pytest.approx(0.71)
    assert torch.allclose(model(sample), restored_model(sample))
    assert restored_optimizer.state_dict()["state"]

    summary = inspect_checkpoint(path)
    assert summary["model_name"] == "tiny_cnn"
    assert summary["has_optimizer_state"] is True
    assert summary["parameter_count"] > 0


def test_checkpoint_compatibility_is_clear_and_allows_epoch_extension(tmp_path: Path) -> None:
    model = create_model("tiny_cnn", pretrained=False)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        epoch=0,
        global_step=1,
        best_metric=0.2,
        class_to_idx=CLASS_TO_IDX,
        model_name="tiny_cnn",
        preprocessing_config=PREPROCESSING,
        training_config={"training": {"lr": 0.01, "epochs": 2}},
    )
    checkpoint = load_checkpoint(path)

    validate_checkpoint_compatibility(
        checkpoint,
        model_name="tiny_cnn",
        class_to_idx=CLASS_TO_IDX,
        preprocessing_config=PREPROCESSING,
        training_config={"training": {"lr": 0.01, "epochs": 10}},
    )
    with pytest.raises(IncompatibleCheckpointError, match="preprocessing"):
        validate_checkpoint_compatibility(
            checkpoint,
            model_name="tiny_cnn",
            class_to_idx=CLASS_TO_IDX,
            preprocessing_config={"image_size": 64, "grayscale": True},
        )
    with pytest.raises(IncompatibleCheckpointError, match="material training"):
        validate_checkpoint_compatibility(
            checkpoint,
            model_name="tiny_cnn",
            class_to_idx=CLASS_TO_IDX,
            preprocessing_config=PREPROCESSING,
            training_config={"training": {"lr": 0.9, "epochs": 2}},
        )
