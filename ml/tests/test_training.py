from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from petcam_ml.models.checkpoint import load_checkpoint
from petcam_ml.train import resolve_device, seed_everything, train
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TinyDataset(TensorDataset):
    def __init__(self) -> None:
        inputs = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.8, 0.2, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.8, 0.2],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        super().__init__(inputs, self.targets)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 3)

    def get_classifier(self) -> nn.Module:
        return self.classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.relu(self.backbone(inputs)))


def training_config(run_dir: Path, epochs: int = 2) -> dict[str, object]:
    return {
        "paths": {"run_dir": str(run_dir)},
        "model": {"name": "tiny-offline", "num_classes": 3, "pretrained": False},
        "preprocessing": {"input_size": [2, 2], "grayscale": True},
        "training": {
            "seed": 7,
            "deterministic": True,
            "device": "cpu",
            "epochs": epochs,
            "batch_size": 3,
            "workers": 0,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "none",
            "mixed_precision": False,
            "class_weights": True,
            "head_warmup_epochs": 1,
        },
        "checkpoint": {
            "monitor": "macro_f1",
            "mode": "max",
            "checkpoint_every_steps": 1,
            "keep_periodic": 2,
        },
    }


def loaders() -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    dataset = TinyDataset()
    return {
        "train": DataLoader(dataset, batch_size=3, shuffle=False),
        "validation": DataLoader(dataset, batch_size=3, shuffle=False),
    }


def test_device_and_seed_are_cpu_safe() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    seed_everything(123)
    first = torch.rand(3)
    seed_everything(123)
    assert torch.equal(first, torch.rand(3))


def test_tiny_offline_training_writes_observable_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "tiny-run"
    summary = train(
        training_config(run_dir),
        fresh=True,
        model=TinyClassifier(),
        dataloaders=loaders(),
    )

    assert summary["status"] == "completed"
    assert summary["device"] == "cpu"
    assert summary["mixed_precision"] is False
    assert summary["global_step"] == 4
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert len(list((run_dir / "checkpoints").glob("periodic-step-*.pt"))) == 2
    assert any((run_dir / "tensorboard").glob("events.out.tfevents.*"))

    checkpoint = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "class_to_idx",
        "preprocessing_config",
        "training_config",
        "epoch",
        "global_step",
        "best_metric",
    ):
        assert key in checkpoint
    assert checkpoint["metadata"]["epoch_complete"] is True
    assert checkpoint["class_to_idx"] == {"standing": 0, "lateral": 1, "unknown": 2}

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {"macro_f1", "lateral_recall", "standing_recall", "unknown_f1"} <= set(rows[0])
    saved_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["status"] == "completed"
    assert saved_summary["latest_validation"]["confusion_matrix"]
