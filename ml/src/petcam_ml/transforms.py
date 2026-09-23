"""Shared image preprocessing for training, evaluation, and inference."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Serializable preprocessing and conservative augmentation settings."""

    input_size: tuple[int, int] = (256, 256)  # height, width
    grayscale: bool = True
    letterbox: bool = True
    letterbox_fill: int = 0
    interpolation: str = "bilinear"
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    horizontal_flip_probability: float = 0.5
    rotation_degrees: float = 5.0
    translation_fraction: float = 0.04
    brightness: float = 0.12
    contrast: float = 0.12
    blur_probability: float = 0.08
    blur_radius: float = 0.7
    noise_probability: float = 0.08
    noise_std: float = 0.01

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "PreprocessConfig":
        if not config:
            return cls()
        values = dict(config)
        augmentation = values.pop("augmentation", values.pop("augmentations", {})) or {}
        if not isinstance(augmentation, Mapping):
            raise TypeError("augmentation configuration must be a mapping")
        for key, value in augmentation.items():
            values.setdefault(key, value)

        aliases = {
            "pad_value": "letterbox_fill",
            "max_rotation_degrees": "rotation_degrees",
            "max_translation_fraction": "translation_fraction",
        }
        for old_name, new_name in aliases.items():
            if old_name in values:
                if new_name in values:
                    raise ValueError(f"Use either {old_name!r} or {new_name!r}, not both")
                values[new_name] = values.pop(old_name)

        size = values.get("input_size", (256, 256))
        if isinstance(size, int):
            values["input_size"] = (size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 2:
            values["input_size"] = (int(size[0]), int(size[1]))
        else:
            raise ValueError("input_size must be an integer or [height, width]")
        if "mean" in values:
            values["mean"] = tuple(float(item) for item in values["mean"])
        if "std" in values:
            values["std"] = tuple(float(item) for item in values["std"])

        known = {field_info.name for field_info in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown preprocessing setting(s): {', '.join(unknown)}")
        result = cls(**values)
        result._validate()
        return result

    def _validate(self) -> None:
        if any(dimension <= 0 for dimension in self.input_size):
            raise ValueError("input_size dimensions must be positive")
        if not 0 <= self.letterbox_fill <= 255:
            raise ValueError("letterbox_fill must be between 0 and 255")
        if len(self.mean) != 3 or len(self.std) != 3 or any(value <= 0 for value in self.std):
            raise ValueError("mean and std must each contain three values; std must be positive")
        for name in (
            "horizontal_flip_probability",
            "blur_probability",
            "noise_probability",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.rotation_degrees < 0 or self.translation_fraction < 0:
            raise ValueError("rotation and translation settings cannot be negative")
        if self.translation_fraction >= 0.5:
            raise ValueError("translation_fraction must be less than 0.5")
        if self.interpolation not in {"nearest", "bilinear", "bicubic", "lanczos"}:
            raise ValueError("interpolation must be nearest, bilinear, bicubic, or lanczos")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_size"] = list(self.input_size)
        payload["mean"] = list(self.mean)
        payload["std"] = list(self.std)
        return payload


def _resample_mode(name: str) -> Image.Resampling:
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }[name]


def letterbox_image(
    image: Image.Image,
    size: tuple[int, int],
    fill: int = 0,
    *,
    interpolation: str = "bilinear",
) -> Image.Image:
    """Resize to fit ``size`` while preserving aspect ratio, then center-pad."""

    target_height, target_width = size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("letterbox size must be positive")
    source_width, source_height = image.size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("input image has invalid dimensions")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, round(source_width * scale)))
    resized_height = max(1, min(target_height, round(source_height * scale)))
    resized = image.resize((resized_width, resized_height), _resample_mode(interpolation))
    fill_color: int | tuple[int, int, int]
    fill_color = fill if resized.mode == "L" else (fill, fill, fill)
    canvas = Image.new(resized.mode, (target_width, target_height), color=fill_color)
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


class ImageTransform:
    """Callable transform with deterministic validation/inference behavior."""

    def __init__(self, config: PreprocessConfig, *, training: bool = False) -> None:
        config._validate()
        self.config = config
        self.training = training

    def _augment(self, image: Image.Image) -> Image.Image:
        config = self.config
        if random.random() < config.horizontal_flip_probability:
            image = ImageOps.mirror(image)
        if config.rotation_degrees:
            angle = random.uniform(-config.rotation_degrees, config.rotation_degrees)
            image = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=config.letterbox_fill,
            )
        if config.translation_fraction:
            width, height = image.size
            dx = random.uniform(-config.translation_fraction, config.translation_fraction) * width
            dy = random.uniform(-config.translation_fraction, config.translation_fraction) * height
            image = image.transform(
                image.size,
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BICUBIC,
                fillcolor=config.letterbox_fill,
            )
        if config.brightness:
            factor = random.uniform(1 - config.brightness, 1 + config.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if config.contrast:
            factor = random.uniform(1 - config.contrast, 1 + config.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if config.blur_radius and random.random() < config.blur_probability:
            radius = random.uniform(0.1, config.blur_radius)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("image transform expects a PIL.Image.Image")
        converted = image.convert("L" if self.config.grayscale else "RGB")
        if self.training:
            converted = self._augment(converted)
        if self.config.letterbox:
            converted = letterbox_image(
                converted,
                self.config.input_size,
                fill=self.config.letterbox_fill,
                interpolation=self.config.interpolation,
            )
        else:
            target_height, target_width = self.config.input_size
            converted = converted.resize(
                (target_width, target_height), _resample_mode(self.config.interpolation)
            )

        array = np.asarray(converted, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.stack((array, array, array), axis=0)
        else:
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if (
            self.training
            and self.config.noise_std
            and random.random() < self.config.noise_probability
        ):
            tensor = (tensor + torch.randn_like(tensor) * self.config.noise_std).clamp_(0, 1)
        mean = torch.tensor(self.config.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.config.std, dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std


def build_transform(
    config: Mapping[str, Any] | PreprocessConfig | None = None, *, training: bool = False
) -> ImageTransform:
    """Build the shared transform used throughout the package."""

    resolved = (
        config if isinstance(config, PreprocessConfig) else PreprocessConfig.from_mapping(config)
    )
    return ImageTransform(resolved, training=training)
