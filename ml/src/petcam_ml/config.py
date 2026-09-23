"""Configuration loading, validation, and reproducible serialization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return payload


def _validate(config: Mapping[str, Any]) -> None:
    required = ("paths", "model", "preprocessing", "data", "training", "checkpoint", "inference")
    missing = [key for key in required if not isinstance(config.get(key), Mapping)]
    if missing:
        raise ValueError(f"Missing configuration section(s): {', '.join(missing)}")

    size = config["preprocessing"].get("input_size")
    if not (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and all(isinstance(item, int) and item > 0 for item in size)
    ):
        raise ValueError("preprocessing.input_size must contain two positive integers")

    fractions = [
        float(config["data"].get(name, -1))
        for name in ("train_fraction", "validation_fraction", "test_fraction")
    ]
    if any(value < 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("data train/validation/test fractions must be non-negative and sum to 1")

    threshold = float(config["inference"].get("confidence_threshold", 0.0))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("inference.confidence_threshold must be between 0 and 1")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults and recursively overlay a YAML configuration file.

    Relative values remain serializable as written. Commands resolve checked-in
    ``configs/`` paths from the ``ml/`` project root.
    """

    defaults = _read_yaml(DEFAULT_CONFIG_PATH)
    selected = Path(path).expanduser() if path is not None else DEFAULT_CONFIG_PATH
    config = (
        defaults
        if selected.resolve() == DEFAULT_CONFIG_PATH.resolve()
        else _deep_merge(defaults, _read_yaml(selected))
    )
    _validate(config)
    config["_config_path"] = str(selected.resolve())
    return config


def save_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a stable YAML snapshot, omitting loader-only private fields."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: deepcopy(value) for key, value in config.items() if not str(key).startswith("_")
    }
    destination.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
    return destination


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the serializable part of a loaded configuration."""

    return {key: deepcopy(value) for key, value in config.items() if not str(key).startswith("_")}


def resolve_config_path(
    value: str | Path,
    config_path: str | Path | None,
) -> Path:
    """Resolve a configured path consistently from any working directory."""

    path = Path(value).expanduser()
    if path.is_absolute() or config_path is None:
        return path.resolve()
    source = Path(config_path).expanduser().resolve()
    # Checked-in configs live under ``ml/configs`` and intentionally express
    # paths from the package/project root.
    base = source.parent.parent if source.parent.name == "configs" else source.parent
    return (base / path).resolve()


try:
    DEFAULT_CONFIG = _read_yaml(DEFAULT_CONFIG_PATH)
except (
    FileNotFoundError,
    ValueError,
):  # pragma: no cover - only useful in malformed source installs
    DEFAULT_CONFIG = {}
