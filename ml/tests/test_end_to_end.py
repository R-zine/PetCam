from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from rabbitcam_ml.__main__ import main
from rabbitcam_ml.inference import Predictor
from rabbitcam_ml.models.factory import create_model
from rabbitcam_ml.train import train


def _config(run_dir: Path, epochs: int) -> dict[str, object]:
    return {
        "paths": {"run_dir": str(run_dir)},
        "model": {"name": "tiny_cnn", "num_classes": 3, "pretrained": False},
        "preprocessing": {
            "input_size": [16, 16],
            "grayscale": True,
            "letterbox": True,
            "pad_value": 0,
            "interpolation": "bilinear",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "training": {
            "seed": 17,
            "deterministic": True,
            "device": "cpu",
            "epochs": epochs,
            "batch_size": 3,
            "workers": 0,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "scheduler": "none",
            "mixed_precision": False,
            "class_weights": False,
            "head_warmup_epochs": 0,
        },
        "checkpoint": {
            "monitor": "macro_f1",
            "mode": "max",
            "checkpoint_every_steps": 2,
            "keep_periodic": 2,
        },
        "inference": {"confidence_threshold": 0.55},
    }


def _loaders() -> dict[str, DataLoader[tuple[torch.Tensor, torch.Tensor]]]:
    generator = torch.Generator().manual_seed(42)
    images = torch.rand((9, 3, 16, 16), generator=generator)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
    dataset = TensorDataset(images, labels)
    return {
        "train": DataLoader(dataset, batch_size=3, shuffle=False),
        "validation": DataLoader(dataset, batch_size=3, shuffle=False),
    }


def test_train_resume_predict_and_cli_json_work_together(
    tmp_path: Path, capsys: object
) -> None:
    run_dir = tmp_path / "portable-tiny-run"
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps(
            {
                "seed": 17,
                "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "session_splits": {
                    "standing-1": "train",
                    "lateral-1": "validation",
                    "unknown-1": "test",
                },
                "samples": [
                    {
                        "path": "synthetic.jpg",
                        "label": "standing",
                        "session_id": "standing-1",
                        "split": "train",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    first_config = _config(run_dir, epochs=1)
    first_config["paths"]["split_manifest"] = str(manifest)  # type: ignore[index]
    first = train(
        first_config,
        fresh=True,
        model=create_model("tiny_cnn", num_classes=3, pretrained=False),
        dataloaders=_loaders(),
    )
    assert first["status"] == "completed"

    resume_config = _config(run_dir, epochs=2)
    resume_config["paths"]["split_manifest"] = str(manifest)  # type: ignore[index]
    resumed = train(
        resume_config,
        model=create_model("tiny_cnn", num_classes=3, pretrained=False),
        dataloaders=_loaders(),
    )
    assert resumed["resumed"] is True
    checkpoint = run_dir / "checkpoints" / "last.pt"
    assert checkpoint.is_file()
    assert (run_dir / "split-manifest.json").read_bytes() == manifest.read_bytes()
    assert (run_dir / "metrics.csv").is_file()
    assert any((run_dir / "tensorboard").glob("events.out.tfevents.*"))

    changed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    changed_manifest["session_splits"]["unknown-1"] = "train"
    manifest.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from the split snapshot"):
        train(
            resume_config,
            model=create_model("tiny_cnn", num_classes=3, pretrained=False),
            dataloaders=_loaders(),
        )

    image_path = tmp_path / "rabbit.jpg"
    Image.fromarray(np.full((12, 20, 3), 127, dtype=np.uint8)).save(image_path)
    prediction = Predictor(checkpoint, device="cpu").predict_path(image_path)
    assert set(prediction.probabilities) == {"standing", "lateral", "unknown"}

    # Discard training summaries before validating the machine-readable command.
    capsys.readouterr()  # type: ignore[attr-defined]
    assert main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(image_path),
            "--device",
            "cpu",
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["label"] in {"standing", "lateral", "unknown"}
    assert payload["raw_label"] in {"standing", "lateral", "unknown"}
    assert set(payload["probabilities"]) == {"standing", "lateral", "unknown"}

    assert main(["inspect-checkpoint", str(checkpoint)]) == 0
    inspection = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert inspection["model_name"] == "tiny_cnn"
    assert inspection["global_step"] == resumed["global_step"]
    assert inspection["metadata"]["data_split"]["session_splits"] == {
        "standing-1": "train",
        "lateral-1": "validation",
        "unknown-1": "test",
    }
