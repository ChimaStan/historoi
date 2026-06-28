"""
Canonical HistoROI class definitions.

The model outputs six logits/probabilities with zero-based indices 0..5.
For user-facing outputs and masks, we expose one-based class IDs 1..6,
leaving 0 available for no prediction / unclassified mask pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


def hex_to_qupath_color_rgb(hex_color: str) -> int:
    """
    Convert '#RRGGBB' to a QuPath-compatible signed ARGB integer.

    QuPath commonly stores annotation colours as signed Java-style integers
    representing 0xAARRGGBB, with alpha set to FF.
    """
    value = hex_color.strip().lstrip("#")

    if len(value) != 6:
        raise ValueError(f"Expected '#RRGGBB', got {hex_color!r}")

    try:
        argb = int(f"FF{value}", 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex colour: {hex_color!r}") from exc

    if argb >= 2**31:
        argb -= 2**32

    return argb


def qupath_color_rgb_to_hex(color_rgb: int) -> str:
    """
    Convert a QuPath-compatible signed ARGB integer to '#RRGGBB'.
    """
    unsigned = int(color_rgb) & 0xFFFFFFFF
    rgb = unsigned & 0xFFFFFF
    return f"#{rgb:06X}"


@dataclass(frozen=True)
class HistoROIClass:
    """
    Definition of one HistoROI output class.

    Attributes
    ----------
    logit_index:
        Zero-based index of the model output logit/probability.
    class_id:
        One-based stable class ID used in masks and downstream processing.
        ID 0 is reserved for no prediction / unclassified pixels.
    name:
        Short label used in CSV and GeoJSON outputs.
    description:
        Longer human-readable class description.
    color_name:
        Human-readable colour name.
    color_hex:
        Canonical colour stored as '#RRGGBB'.
    """

    logit_index: int
    class_id: int
    name: str
    description: str
    color_name: str
    color_hex: str

    @property
    def color_rgb(self) -> int:
        """QuPath-compatible signed ARGB colour integer."""
        return hex_to_qupath_color_rgb(self.color_hex)


HISTOROI_CLASSES: Final[tuple[HistoROIClass, ...]] = (
    HistoROIClass(
        logit_index=0,
        class_id=1,
        name="Epithelial",
        description="Epithelial region",
        color_name="cyan",
        color_hex="#00FFFF",
    ),
    HistoROIClass(
        logit_index=1,
        class_id=2,
        name="Stroma",
        description="Stromal region",
        color_name="blue",
        color_hex="#0000FF",
    ),
    HistoROIClass(
        logit_index=2,
        class_id=3,
        name="Adipose",
        description="Adipose / Scattered stroma",
        color_name="magenta",
        color_hex="#FF00FF",
    ),
    HistoROIClass(
        logit_index=3,
        class_id=4,
        name="Artefact",
        description="Artefacts",
        color_name="red",
        color_hex="#FF0000",
    ),
    HistoROIClass(
        logit_index=4,
        class_id=5,
        name="Miscellaneous",
        description="Miscellaneous",
        color_name="black",
        color_hex="#000000",
    ),
    HistoROIClass(
        logit_index=5,
        class_id=6,
        name="Lymphocytes",
        description="Lymphocyte dense region",
        color_name="dark green",
        color_hex="#1A4D1A",
    ),
)


def validate_class_definitions() -> None:
    """Validate that class definitions are internally consistent."""
    model_indices = [cls.logit_index for cls in HISTOROI_CLASSES]
    class_ids = [cls.class_id for cls in HISTOROI_CLASSES]
    names = [cls.name for cls in HISTOROI_CLASSES]

    expected_model_indices = list(range(len(HISTOROI_CLASSES)))
    expected_class_ids = list(range(1, len(HISTOROI_CLASSES) + 1))

    if model_indices != expected_model_indices:
        raise ValueError(
            f"Model indices must be consecutive 0-based values. "
            f"Expected {expected_model_indices}, got {model_indices}."
        )

    if class_ids != expected_class_ids:
        raise ValueError(
            f"Class IDs must be consecutive 1-based values. "
            f"Expected {expected_class_ids}, got {class_ids}."
        )

    if len(set(names)) != len(names):
        raise ValueError(f"Class names must be unique, got {names}.")

    for cls in HISTOROI_CLASSES:
        recovered_hex = qupath_color_rgb_to_hex(cls.color_rgb)
        if recovered_hex != cls.color_hex.upper():
            raise ValueError(
                f"Colour round-trip failed for {cls.name!r}: "
                f"{cls.color_hex!r} -> {cls.color_rgb!r} -> {recovered_hex!r}."
            )


validate_class_definitions()


CLASS_BY_ID: Final[dict[int, HistoROIClass]] = {
    cls.class_id: cls for cls in HISTOROI_CLASSES
}

CLASS_BY_NAME: Final[dict[str, HistoROIClass]] = {
    cls.name: cls for cls in HISTOROI_CLASSES
}


LOGIT_INDEX_TO_NAME: Final[dict[int, str]] = {
    cls.logit_index: cls.name for cls in HISTOROI_CLASSES
}

LOGIT_INDEX_TO_CLASS_ID: Final[dict[int, int]] = {
    cls.logit_index: cls.class_id for cls in HISTOROI_CLASSES
}


CLASS_COLORS_HEX: Final[dict[str, str]] = {
    cls.name: cls.color_hex for cls in HISTOROI_CLASSES
}

CLASS_COLOR_NAMES: Final[dict[str, str]] = {
    cls.name: cls.color_name for cls in HISTOROI_CLASSES
}

CLASS_COLORS_RGB: Final[dict[str, int]] = {
    cls.name: cls.color_rgb for cls in HISTOROI_CLASSES
}

