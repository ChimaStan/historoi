"""
HistoROI whole-slide inference pipeline.

This module performs inference on one or more whole-slide images using
a trained HistoROI six-class classifier. Candidate tissue locations are first
identified (optionally using a tissue mask), extracted as image patches, and
classified by the classifier. Predictions are written to disk together
with optional visualisation and post-processing outputs.

Primary outputs
---------------
For each processed WSI:

- ``csvs/<slide>.csv``
    Patch-level prediction table containing:
    - level-0 patch centre coordinates,
    - predicted class ID,
    - predicted class name,
    - per-class probabilities.

- ``csvs/historoi_prediction_summary.csv``
    One-row-per-slide summary of predicted class distributions, including
    total patches, per-class counts, and per-class percentages.

- ``logs/inference.log``
    Human-readable inference log.

- ``logs/run_config.json``
    Configuration used for the inference run.

If one or more slides fail during processing:

- ``logs/inference_errors.csv``
    Summary of failed slides and corresponding exceptions.

Optional outputs
----------------
Depending on the selected command-line options:

- ``visualizations/*.geojson``
    Legacy point-based prediction visualisations.

- ``annotations/*_historoi_pred_polygons.geojson``
    Patch-footprint polygon annotations suitable for downstream tools.

- ``masks/*_historoi_pred_mask.png``
    Downsampled prediction masks encoded using stable HistoROI class IDs.
"""

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openslide import OpenSlide
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from args import infer_options
from class_labels import LOGIT_INDEX_TO_CLASS_ID, LOGIT_INDEX_TO_NAME
from dataset import WSIDataset
from postprocessing import export_pred_polygons_and_masks, pred_summary, save_pred_summary
from utils import filtered_patches, find_matching_mask, get_metadata, load_image_mask
from utils import resolve_wsi_paths, resolve_device, resolve_mask_paths
from visualization import make_geojson


def setup_logging(log_dir: str | Path) -> logging.Logger:
    """Configure inference logging to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("historoi.inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_dir / "inference.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main() -> None:
    args = infer_options()
    device = resolve_device(args.device)

    model_path = Path(args.model6)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model6 = models.resnet18().to(device)
    model6.fc = torch.nn.Linear(512, 6).to(device)
    model6.load_state_dict(torch.load(model_path, map_location=device))
    model6.eval()

    sm = torch.nn.Softmax(dim=1)
    bs = args.batch_size

    output_dir = Path(args.output_dir)
    csv_dir = output_dir / "csvs"
    vis_dir = output_dir / "visualizations"
    log_dir = output_dir / "logs"

    annotations_dir = output_dir / "annotations"
    masks_dir = output_dir / "masks"

    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.export_pred_polygons:
        annotations_dir.mkdir(parents=True, exist_ok=True)

    if args.export_pred_mask:
        masks_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir)

    config_path = log_dir / "run_config.json"
    wsi_paths = resolve_wsi_paths(args.wsis)
    mask_paths = resolve_mask_paths(args.mask_path)

    if mask_paths is not None and len(mask_paths) == 1 and len(wsi_paths) != 1:
        raise ValueError(
            "--mask_path points to a single mask file, but multiple WSIs were found. "
            "Use a mask directory or wildcard pattern for batch inference."
        )

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    logger.info("Saved run configuration: %s", config_path)
    logger.info("Run configuration: %s", vars(args))
    logger.info("Found %d WSI(s) for inference.", len(wsi_paths))

    summary_rows = []
    failures = []

    for w in tqdm(wsi_paths):
        wsi = None

        try:
            logger.info("Processing WSI: %s", w)

            wsi = OpenSlide(w)
            csv_path = csv_dir / f"{Path(w).stem}.csv"

            if args.skip_if_present and csv_path.is_file():
                logger.info("Skipping existing output: %s", csv_path)
                continue

            level, ps, magni_0, ps_level0 = get_metadata(wsi, args)
            half_ps_level0 = ps_level0 // 2
            stride_level0 = round(args.stride * (magni_0 / 10))

            mask = None
            mask_path = find_matching_mask(
                wsi_path=w,
                mask_paths=mask_paths,
                require_mask=args.require_mask,
            )

            if mask_path is not None:
                logger.info("Using mask: %s", mask_path)
                mask = load_image_mask(mask_path)

            df = filtered_patches(
                wsi=wsi,
                stride_level0=stride_level0,
                mask=mask,
                mask_labels=args.mask_labels,
            )

            if len(df) == 0:
                logger.warning("No candidate patches found for %s; skipping.", w)
                continue

            logger.info("Number of candidate patches: %d", len(df))
            logger.info("Number of batches: %d", 1 + len(df) // args.batch_size)

            base_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
            ])

            ds = WSIDataset(
                df=df,
                wsi=wsi,
                transform=base_transform,
                level=level,
                ps=ps,
                half_ps_level0=half_ps_level0,
            )

            dl = DataLoader(
                ds,
                batch_size=bs,
                shuffle=False,
                num_workers=args.workers,
            )

            probs = np.zeros((len(ds), 6))

            with torch.no_grad():
                for i, data in tqdm(enumerate(dl)):
                    out = model6(data.to(device))
                    probs[bs * i:bs * i + data.shape[0], :6] = sm(out).cpu().numpy()

            pred_index = np.argmax(probs, axis=1).astype(int)
            df["class_id"] = [LOGIT_INDEX_TO_CLASS_ID[i] for i in pred_index]
            df["preds"] = [LOGIT_INDEX_TO_NAME[i] for i in pred_index]

            for idx, name in LOGIT_INDEX_TO_NAME.items():
                df[f"prob_{name}"] = probs[:, idx]

            df.to_csv(csv_path, index=False)
            logger.info("Saved prediction CSV: %s", csv_path)

            summary_rows.append(
                pred_summary(slide_stem=Path(w).stem, df=df)
            )

            if args.vis:
                geojson_path = make_geojson(csv_path, vis_dir)
                logger.info("Saved legacy visualisation GeoJSON: %s", geojson_path)

            if args.export_pred_polygons or args.export_pred_mask:
                artifact_paths = export_pred_polygons_and_masks(
                    csv_path=csv_path,
                    slide_dims=wsi.dimensions,
                    output_dir=output_dir,
                    slide_stem=Path(w).stem,
                    ps_level0=ps_level0,
                    stride_level0=stride_level0,
                    export_pred_polygons=args.export_pred_polygons,
                    export_pred_mask=args.export_pred_mask,
                    mask_ds=args.mask_ds,
                    overlap_policy=args.overlap_policy,
                )

                for artifact_name, artifact_path in artifact_paths.items():
                    logger.info("Saved %s: %s", artifact_name, artifact_path)
                                    
        except Exception as exc:
            logger.exception("Failed processing WSI: %s", w)

            failures.append({
                "wsi": w,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })

            continue

        finally:
            if wsi is not None:
                wsi.close()

        summary_path = save_pred_summary(
            summary_rows=summary_rows,
            output_path=csv_dir / "historoi_prediction_summary.csv",
        )

        if summary_rows:
            logger.info("Saved prediction summary: %s", summary_path)

    if failures:
        error_path = log_dir / "inference_errors.csv"
        pd.DataFrame(failures).to_csv(error_path, index=False)
        logger.info("Saved inference error summary: %s", error_path)


if __name__ == "__main__":
    main()