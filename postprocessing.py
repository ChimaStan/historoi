"""
Post-processing utilities for HistoROI prediction outputs.

This module converts HistoROI patch-centre predictions into reusable
prediction polygon GeoJSON annotations and optional downsampled prediction
class masks, usable by external packages such as PySlyde.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from class_labels import (
    CLASS_BY_ID,
    CLASS_COLORS_RGB,
    HISTOROI_CLASSES,
)

VALID_OVERLAP_POLICIES = {
    "error",
    "allow_raw",
    "retain_first",
    "retain_last",
    "highest_prob",
}


def _class_metadata() -> list[dict[str, Any]]:
    """Return class metadata in a JSON-friendly structure."""
    return [
        {
            "class_id": int(cls.class_id),
            "name": cls.name,
            "description": cls.description,
            "color_name": cls.color_name,
            "color_hex": cls.color_hex,
            "color_rgb": int(cls.color_rgb),
        }
        for cls in HISTOROI_CLASSES
    ]


def _has_patch_overlap(
    *,
    ps_level0: int,
    stride_level0: int,
) -> bool:
    """
    Return whether neighbouring HistoROI patch footprints overlap.
    """
    if ps_level0 <= 0:
        raise ValueError(f"Invalid ps_level0: {ps_level0}")

    if stride_level0 <= 0:
        raise ValueError(f"Invalid stride_level0: {stride_level0}")

    return stride_level0 < ps_level0


def _validate_requested_exports(
    *,
    export_pred_polygons: bool,
    export_pred_mask: bool,
    overlap_policy: str,
    has_overlap: bool,
) -> None:
    """
    Validate that the requested outputs and overlap policy are compatible.

    This prevents partially writing one output before discovering that another
    requested output cannot honestly represent the requested overlap policy.
    """
    if overlap_policy not in VALID_OVERLAP_POLICIES:
        raise ValueError(
            f"Unsupported overlap_policy={overlap_policy!r}. "
            f"Use one of: {sorted(VALID_OVERLAP_POLICIES)}."
        )

    if not export_pred_polygons and not export_pred_mask:
        return

    if not has_overlap:
        return

    if overlap_policy == "error":
        raise ValueError(
            "Overlapping HistoROI patch footprints were detected. "
            "Use --stride 256 to avoid overlap, or choose an explicit "
            "--overlap_policy suitable for the requested output."
        )

    if export_pred_polygons and overlap_policy != "allow_raw":
        raise ValueError(
            f"Cannot export overlapping polygon GeoJSON with "
            f"overlap_policy={overlap_policy!r}. Polygon GeoJSON would still "
            "contain overlapping raw patch annotations, so mask-resolution "
            "policies such as 'retain_first', 'retain_last', and "
            "'highest_prob' cannot be represented honestly as polygons. "
            "Use --overlap_policy allow_raw for raw overlapping polygons, "
            "or disable polygon export and export a resolved prediction mask."
        )

    if export_pred_mask and overlap_policy == "allow_raw":
        raise ValueError(
            "Cannot export a prediction mask with overlap_policy='allow_raw'. "
            "'allow_raw' is only meaningful for raw overlapping polygon GeoJSON. "
            "For masks, use 'retain_first', 'retain_last', or 'highest_prob'."
        )


def _read_pred_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Read and validate a HistoROI prediction CSV.

    Expected columns:
        dim1:
            X-coordinate of patch centre in level-0 WSI coordinates.
        dim2:
            Y-coordinate of patch centre in level-0 WSI coordinates.
        class_id:
            Stable one-based HistoROI class ID. ID 0 is reserved for
            no prediction / unclassified pixels in masks.
        preds:
            Canonical HistoROI class name.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"dim1", "dim2", "class_id", "preds"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction CSV must contain columns {sorted(required)}. "
            f"Missing: {sorted(missing)}. If this is an old HistoROI CSV, "
            "rerun inference with the updated class-label export."
        )

    df = df.copy()

    for coord_col in ("dim1", "dim2"):
        df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")

        if df[coord_col].isna().any():
            bad_rows = df[df[coord_col].isna()].index.tolist()[:10]
            raise ValueError(
                f"Column {coord_col!r} contains non-numeric values. "
                f"Example bad row indices: {bad_rows}"
            )

    df["class_id"] = pd.to_numeric(df["class_id"], errors="coerce")

    if df["class_id"].isna().any():
        bad_rows = df[df["class_id"].isna()].index.tolist()[:10]
        raise ValueError(
            "Column 'class_id' contains non-numeric values. "
            f"Example bad row indices: {bad_rows}"
        )

    df["class_id"] = df["class_id"].astype(int)
    df["preds"] = df["preds"].astype(str).str.strip()

    valid_ids = set(CLASS_BY_ID)
    observed_ids = set(df["class_id"].unique())
    invalid_ids = sorted(observed_ids - valid_ids)

    if invalid_ids:
        raise ValueError(
            f"Unknown HistoROI class_id value(s): {invalid_ids}. "
            f"Expected one of: {sorted(valid_ids)}"
        )

    expected_names = df["class_id"].map(
        lambda class_id: CLASS_BY_ID[int(class_id)].name
    )

    mismatched = df["preds"] != expected_names

    if mismatched.any():
        examples = df.loc[mismatched, ["class_id", "preds"]].head(10)
        raise ValueError(
            "Mismatch between 'class_id' and 'preds' in prediction CSV. "
            "Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    return df


def _patch_polygon_bounds(
    cx: float,
    cy: float,
    ps_level0: int,
    slide_width: int,
    slide_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert a HistoROI patch-centre coordinate into clipped level-0
    patch-footprint polygon bounds.

    Returns:
        (x0, y0, x1, y1), using half-open coordinates [x0, x1), [y0, y1).
    """
    half = ps_level0 / 2.0

    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x1 = int(round(cx + half))
    y1 = int(round(cy + half))

    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, int(slide_width))
    y1 = min(y1, int(slide_height))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Invalid polygon bounds from centre ({cx}, {cy}): "
            f"({x0}, {y0}, {x1}, {y1})"
        )

    return x0, y0, x1, y1


def _patch_polygon_feature(
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    class_id: int,
    label: str,
    cx: int,
    cy: int,
    polygon_id: int,
) -> dict[str, Any]:
    """
    Create a GeoJSON polygon Feature for one rectangular HistoROI patch region.
    """
    coords = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
        [x0, y0],
    ]

    return {
        "type": "Feature",
        "properties": {
            "label": label,
            "class_id": int(class_id),
            "classification": {
                "name": label,
                "colorRGB": int(CLASS_COLORS_RGB[label]),
            },
            "polygon_id": int(polygon_id),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def _validate_probability_columns(df: pd.DataFrame) -> None:
    """
    Validate that probability columns exist for all predicted class labels.

    Expected column naming convention:
        prob_<class name>

    For example:
        prob_Epithelial
        prob_Stroma
    """
    labels = sorted(df["preds"].astype(str).unique())
    required_prob_cols = [f"prob_{label}" for label in labels]

    missing = [col for col in required_prob_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "overlap_policy='highest_prob' requires probability columns in the "
            f"prediction CSV. Missing columns: {missing}"
        )

    for col in required_prob_cols:
        values = pd.to_numeric(df[col], errors="coerce")

        if values.isna().any():
            bad_rows = df[values.isna()].index.tolist()[:10]
            raise ValueError(
                f"Probability column {col!r} contains non-numeric values. "
                f"Example bad row indices: {bad_rows}"
            )

        df[col] = values.astype(float)


def preds_to_patch_geojson(
    *,
    csv_path: str | Path,
    slide_dims: tuple[int, int],
    output_path: str | Path,
    ps_level0,
    stride_level0,
    overlap_policy: str = "error",
) -> Path:
    """
    Export HistoROI predictions as rectangular patch-footprint polygon GeoJSON.

    The output polygons are expressed in level-0 WSI coordinates and use
    ``properties["label"]`` for compatibility with PySlyde's GeoJSON loader.

    If patch footprints overlap, only overlap_policy='allow_raw' can export
    polygon GeoJSON. Mask-resolution policies cannot be represented honestly
    as raw patch-footprint polygons.
    """
    if overlap_policy not in VALID_OVERLAP_POLICIES:
        raise ValueError(
            f"Unsupported overlap_policy={overlap_policy!r}. "
            f"Use one of: {sorted(VALID_OVERLAP_POLICIES)}."
        )

    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide_width, slide_height = map(int, slide_dims)

    has_overlap = _has_patch_overlap(
        ps_level0=ps_level0,
        stride_level0=stride_level0,
    )

    if has_overlap and overlap_policy == "error":
        raise ValueError(
            "Cannot export patch-footprint polygon GeoJSON because HistoROI "
            "patch footprints overlap. Use --stride 256, or set "
            "--overlap_policy allow_raw if you intentionally want raw "
            "overlapping polygon annotations."
        )

    if has_overlap and overlap_policy != "allow_raw":
        raise ValueError(
            f"Cannot export overlapping polygon GeoJSON with "
            f"overlap_policy={overlap_policy!r}. Use --overlap_policy allow_raw "
            "for raw overlapping polygons, or disable polygon export and export "
            "a resolved prediction mask instead."
        )

    df = _read_pred_csv(csv_path)

    features: list[dict[str, Any]] = []

    for polygon_id, row in enumerate(df.itertuples(index=False), start=1):
        cx = int(round(float(getattr(row, "dim1"))))
        cy = int(round(float(getattr(row, "dim2"))))
        class_id = int(getattr(row, "class_id"))
        label = str(getattr(row, "preds"))

        x0, y0, x1, y1 = _patch_polygon_bounds(
            cx=cx,
            cy=cy,
            ps_level0=ps_level0,
            slide_width=slide_width,
            slide_height=slide_height,
        )

        features.append(
            _patch_polygon_feature(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                class_id=class_id,
                label=label,
                cx=cx,
                cy=cy,
                polygon_id=polygon_id,
            )
        )

    metadata: dict[str, Any] = {
        "source": "HistoROI",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_space": "level-0",
        "slide_dimensions_level0": [slide_width, slide_height],
        "patch_size_level0": ps_level0,
        "stride_level0": stride_level0,
        "overlapping_patch_footprints": has_overlap,
        "overlap_policy": overlap_policy,
        "classes": _class_metadata(),
    }

    if has_overlap and overlap_policy == "allow_raw":
        metadata["warning"] = (
            "This GeoJSON contains overlapping patch-footprint polygons. "
            "Downstream rasterisation may be ambiguous unless the consuming "
            "tool handles overlapping annotations explicitly."
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": metadata,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    return output_path


def preds_to_class_mask_png(
    *,
    csv_path: str | Path,
    slide_dims: tuple[int, int],
    output_path: str | Path,
    ps_level0,
    stride_level0,
    downsample: float = 16.0,
    overlap_policy: str = "error",
) -> Path:
    """
    Export HistoROI predictions as a downsampled prediction mask PNG.

    Pixel values:
        0 = no prediction / unclassified
        1..6 = HistoROI class IDs

    Args:
        downsample:
            Downsample factor relative to level 0. Use 1 for level-0 output.
            Larger values produce smaller masks.
        overlap_policy:
            Behaviour when patch footprints overlap:
            - "error": raise if stride is smaller than patch size
            - "retain_first": keep the first prediction assigned to each pixel
            - "retain_last": let later predictions overwrite earlier predictions
            - "highest_prob": keep the prediction with the highest probability
              for its predicted class
    """
    if overlap_policy not in VALID_OVERLAP_POLICIES:
        raise ValueError(
            f"Unsupported overlap_policy={overlap_policy!r}. "
            f"Use one of: {sorted(VALID_OVERLAP_POLICIES)}."
        )

    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")

    csv_path = Path(csv_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    slide_width, slide_height = map(int, slide_dims)

    has_overlap = _has_patch_overlap(
        ps_level0=ps_level0,
        stride_level0=stride_level0,
    )

    if has_overlap and overlap_policy == "error":
        raise ValueError(
            "Cannot export a prediction mask because HistoROI patch "
            "footprints overlap. Use --stride 256, or set --overlap_policy "
            "to 'retain_first', 'retain_last', or 'highest_prob'."
        )

    if has_overlap and overlap_policy == "allow_raw":
        raise ValueError(
            "Cannot export a prediction mask with overlap_policy='allow_raw'. "
            "'allow_raw' is only valid for raw overlapping polygon GeoJSON. "
            "For masks, use 'retain_first', 'retain_last', or 'highest_prob'."
        )

    mask_width = int(math.ceil(slide_width / downsample))
    mask_height = int(math.ceil(slide_height / downsample))

    if mask_width <= 0 or mask_height <= 0:
        raise ValueError(
            f"Invalid mask dimensions: {(mask_width, mask_height)} from "
            f"slide_dims={slide_dims}, downsample={downsample}"
        )

    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    df = _read_pred_csv(csv_path)

    score_mask: np.ndarray | None = None

    if has_overlap and overlap_policy == "highest_prob":
        _validate_probability_columns(df)
        score_mask = np.full((mask_height, mask_width), -np.inf, dtype=np.float32)

    for _, row in df.iterrows():
        cx = int(round(float(row["dim1"])))
        cy = int(round(float(row["dim2"])))
        class_id = int(row["class_id"])
        label = str(row["preds"])

        x0, y0, x1, y1 = _patch_polygon_bounds(
            cx=cx,
            cy=cy,
            ps_level0=ps_level0,
            slide_width=slide_width,
            slide_height=slide_height,
        )

        mx0 = max(int(math.floor(x0 / downsample)), 0)
        my0 = max(int(math.floor(y0 / downsample)), 0)
        mx1 = min(int(math.ceil(x1 / downsample)), mask_width)
        my1 = min(int(math.ceil(y1 / downsample)), mask_height)

        if mx1 <= mx0 or my1 <= my0:
            continue

        if not has_overlap:
            mask[my0:my1, mx0:mx1] = class_id

        elif overlap_policy == "retain_last":
            mask[my0:my1, mx0:mx1] = class_id

        elif overlap_policy == "retain_first":
            mask_region = mask[my0:my1, mx0:mx1]
            unassigned = mask_region == 0
            mask_region[unassigned] = class_id

        elif overlap_policy == "highest_prob":
            if score_mask is None:
                raise RuntimeError("score_mask was not initialised.")

            prob_col = f"prob_{label}"
            score = float(row[prob_col])

            score_region = score_mask[my0:my1, mx0:mx1]
            mask_region = mask[my0:my1, mx0:mx1]

            update = score > score_region

            mask_region[update] = class_id
            score_region[update] = score

        else:
            raise RuntimeError(f"Unexpected overlap_policy: {overlap_policy!r}")

    Image.fromarray(mask).save(output_path)

    return output_path


def export_pred_polygons_and_masks(
    *,
    csv_path: str | Path,
    slide_dims: tuple[int, int],
    output_dir: str | Path,
    slide_stem: str,
    ps_level0,
    stride_level0,
    export_pred_polygons: bool = False,
    export_pred_mask: bool = False,
    mask_ds: float = 16.0,
    overlap_policy: str = "error",
) -> dict[str, Path]:
    """
    Export prediction-derived polygon and/or mask outputs for one slide.

    Outputs:
        - annotations/<slide_stem>_historoi_pred_polygons.geojson
        - masks/<slide_stem>_historoi_pred_mask.png
    """
    has_overlap = _has_patch_overlap(
        ps_level0=ps_level0,
        stride_level0=stride_level0,
    )

    _validate_requested_exports(
        export_pred_polygons=export_pred_polygons,
        export_pred_mask=export_pred_mask,
        overlap_policy=overlap_policy,
        has_overlap=has_overlap,
    )

    output_dir = Path(output_dir)

    paths: dict[str, Path] = {}

    if export_pred_polygons:
        geojson_path = (
            output_dir / "annotations" / f"{slide_stem}_historoi_pred_polygons.geojson"
        )

        paths["pred_polygon_geojson"] = preds_to_patch_geojson(
            csv_path=csv_path,
            slide_dims=slide_dims,
            output_path=geojson_path,
            ps_level0=ps_level0,
            stride_level0=stride_level0,
            overlap_policy=overlap_policy,
        )

    if export_pred_mask:
        mask_path = output_dir / "masks" / f"{slide_stem}_historoi_pred_mask.png"
   
        mask_png = preds_to_class_mask_png(
            csv_path=csv_path,
            slide_dims=slide_dims,
            output_path=mask_path,
            ps_level0=ps_level0,
            stride_level0=stride_level0,
            downsample=mask_ds,
            overlap_policy=overlap_policy,
        )

        paths["pred_mask_png"] = mask_png

    return paths


def pred_summary(
    *,
    slide_stem: str,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Summarise HistoROI predictions for a single whole-slide image.

    The returned dictionary contains the total number of predicted patches,
    together with the count and percentage of patches assigned to each
    HistoROI class.

    Parameters
    ----------
    slide_stem
        Stem of the WSI filename.

    df
        Prediction dataframe produced during inference.

    Returns
    -------
    dict[str, Any]
        One summary row suitable for inclusion in the run-level prediction
        summary CSV.
    """
    counts = (
        df["preds"]
        .value_counts()
        .reindex(
            [cls.name for cls in HISTOROI_CLASSES],
            fill_value=0,
        )
    )

    total = int(len(df))

    summary: dict[str, Any] = {
        "WSI": slide_stem,
        "Total patches": total,
    }

    for cls in HISTOROI_CLASSES:
        count = int(counts[cls.name])

        summary[cls.name] = count
        summary[f"{cls.name}_pct"] = (
            round(100.0 * count / total, 2)
            if total > 0
            else 0.0
        )

    return summary


def save_pred_summary(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path | None:
    """
    Save the prediction summary for all processed WSIs.

    Parameters
    ----------
    summary_rows
        List of summary rows returned by ``prediction_summary_row()``.

    output_path
        Destination CSV file.

    Returns
    -------
    Path
        Path to the written CSV.

    Notes
    -----
    If no summary rows are supplied, no file is written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        return None

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(output_path, index=False)

    return output_path