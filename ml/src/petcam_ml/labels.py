"""Canonical labels and validation helpers for PetCam ML."""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Final


class Label(StrEnum):
    """The three posture classes produced by the model."""

    STANDING = "standing"
    LATERAL = "lateral"
    UNKNOWN = "unknown"


class Lighting(StrEnum):
    """Lighting metadata recorded for capture sessions."""

    VISIBLE = "visible"
    IR = "ir"
    MIXED = "mixed"
    UNKNOWN = "unknown"


CLASS_NAMES: Final[tuple[str, ...]] = tuple(label.value for label in Label)
CLASS_TO_INDEX = MappingProxyType({name: index for index, name in enumerate(CLASS_NAMES)})
INDEX_TO_CLASS = MappingProxyType({index: name for name, index in CLASS_TO_INDEX.items()})
LIGHTING_MODES: Final[tuple[str, ...]] = tuple(mode.value for mode in Lighting)


def validate_label(value: str | Label) -> str:
    """Return a canonical label or raise a descriptive :class:`ValueError`."""

    candidate = value.value if isinstance(value, Label) else str(value).strip().lower()
    if candidate not in CLASS_TO_INDEX:
        choices = ", ".join(CLASS_NAMES)
        raise ValueError(f"Invalid label {value!r}; expected one of: {choices}")
    return candidate


def validate_lighting(value: str | Lighting) -> str:
    """Return a canonical lighting mode or raise a descriptive error."""

    candidate = value.value if isinstance(value, Lighting) else str(value).strip().lower()
    if candidate not in LIGHTING_MODES:
        choices = ", ".join(LIGHTING_MODES)
        raise ValueError(f"Invalid lighting mode {value!r}; expected one of: {choices}")
    return candidate


def class_mapping() -> dict[str, int]:
    """Return a serializable copy of the stable class-to-index mapping."""

    return dict(CLASS_TO_INDEX)
