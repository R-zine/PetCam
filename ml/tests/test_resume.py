from __future__ import annotations

from pathlib import Path

import pytest
import torch
from petcam_ml.models.checkpoint import IncompatibleCheckpointError, load_checkpoint
from petcam_ml.train import train
from test_training import TinyClassifier, loaders, training_config


class InterruptOnSecondForward(TinyClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 2:
            raise KeyboardInterrupt
        return super().forward(inputs)


def test_normal_invocation_auto_resumes_last_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "resume-run"
    first = train(
        training_config(run_dir, epochs=1),
        fresh=True,
        model=TinyClassifier(),
        dataloaders=loaders(),
    )
    assert first["global_step"] == 2

    resumed = train(
        training_config(run_dir, epochs=2),
        model=TinyClassifier(),
        dataloaders=loaders(),
    )
    assert resumed["resumed"] is True
    assert resumed["epoch"] == 2
    assert resumed["global_step"] == 4
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert checkpoint["epoch"] == 1
    assert checkpoint["metadata"]["epoch_complete"] is True


def test_partial_epoch_checkpoint_resumes_from_next_batch(tmp_path: Path) -> None:
    run_dir = tmp_path / "interrupted-run"
    config = training_config(run_dir, epochs=1)
    config["checkpoint"]["checkpoint_every_steps"] = 0  # type: ignore[index]
    with pytest.raises(KeyboardInterrupt):
        train(
            config,
            fresh=True,
            model=InterruptOnSecondForward(),
            dataloaders=loaders(),
        )

    interrupted = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    assert interrupted["global_step"] == 1
    assert interrupted["metadata"]["epoch_complete"] is False
    assert interrupted["metadata"]["batch_in_epoch"] == 1

    resumed = train(config, model=TinyClassifier(), dataloaders=loaders())
    assert resumed["resumed"] is True
    assert resumed["global_step"] == 2
    assert resumed["status"] == "completed"


def test_auto_resume_rejects_materially_incompatible_preprocessing(tmp_path: Path) -> None:
    run_dir = tmp_path / "incompatible-run"
    config = training_config(run_dir, epochs=1)
    train(config, fresh=True, model=TinyClassifier(), dataloaders=loaders())
    changed = training_config(run_dir, epochs=2)
    changed["preprocessing"] = {"input_size": [99, 99], "grayscale": True}

    with pytest.raises(IncompatibleCheckpointError, match="preprocessing"):
        train(changed, model=TinyClassifier(), dataloaders=loaders())
