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
from dataset import WSIDataset
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

    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

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

            names = [
                "Epithelial",
                "Stroma",
                "Adipose",
                "Artefact",
                "Miscelleneous",
                "Lymphocytes",
            ]

            maps = {idx: name for idx, name in enumerate(names)}

            df["preds"] = np.argmax(probs, axis=1)
            df["preds"] = df["preds"].map(maps)

            df.to_csv(csv_path, index=False)

            if args.vis:
                make_geojson(csv_path, vis_dir)
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

    if failures:
        error_path = log_dir / "inference_errors.csv"
        pd.DataFrame(failures).to_csv(error_path, index=False)
        logger.info("Saved inference error summary: %s", error_path)


if __name__ == "__main__":
    main()