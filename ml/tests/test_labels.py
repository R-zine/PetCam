from __future__ import annotations

import pytest

from rabbitcam_ml.labels import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    LIGHTING_MODES,
    class_mapping,
    validate_label,
    validate_lighting,
)


def test_canonical_label_mapping_is_stable() -> None:
    assert CLASS_NAMES == ("standing", "lateral", "unknown")
    assert class_mapping() == {"standing": 0, "lateral": 1, "unknown": 2}
    assert dict(CLASS_TO_INDEX) == class_mapping()


@pytest.mark.parametrize("value", ["standing", " LATERAL ", "unknown"])
def test_label_validation_returns_canonical_value(value: str) -> None:
    assert validate_label(value) == value.strip().lower()


def test_invalid_label_is_rejected_with_choices() -> None:
    with pytest.raises(ValueError, match="standing, lateral, unknown"):
        validate_label("sleeping")


def test_lighting_validation() -> None:
    assert LIGHTING_MODES == ("visible", "ir", "mixed", "unknown")
    assert validate_lighting(" IR ") == "ir"
    with pytest.raises(ValueError, match="Invalid lighting"):
        validate_lighting("night")
