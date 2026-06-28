# historoi

## Fork notice

This repository is a fork of the original [HistoROI](https://github.com/abhijeetptl5/historoi) repository. It has been developed into an inference-focused pipeline with improved output organisation, structured logging, containerised installation, optional mask-constrained inference, prediction summary reporting, and optional export of prediction polygon annotations and prediction masks.

The underlying HistoROI model architecture and six tissue classes remain unchanged.

## HistoROI classes

HistoROI predicts one of six tissue classes for every analysed image patch:

| Class ID | Tissue class               |
| -------: | -------------------------- |
|        1 | Epithelial                 |
|        2 | Stroma                     |
|        3 | Adipose / Scattered stroma |
|        4 | Artefact                   |
|        5 | Miscellaneous              |
|        6 | Lymphocytes                |

# Requirements

## Local installation

* Conda or Mamba
* Python 3.10
* OpenSlide support (installed through the supplied `environment.yml`)
* Sufficient RAM and storage for whole-slide image inference

## Containerised installation

* Docker, Singularity or Apptainer
* Internet access during image build (unless using a pre-built image)
* Sufficient storage for the container image

## GPU inference

CUDA inference is currently supported only for NVIDIA GPUs.

Requirements:

* NVIDIA GPU
* Compatible NVIDIA drivers
* Docker: NVIDIA Container Toolkit (`--gpus all`)
* Singularity / Apptainer: `--nv`

CPU inference is automatically used when CUDA is unavailable or when `--device cpu` is specified.

# Installation

## Local installation

```bash
git clone https://github.com/chimastan/historoi.git
cd historoi

conda env create -f environment.yml
conda activate historoi
```

## Containerised installation (recommended)

Containerised execution is recommended for reproducible inference, particularly on HPC systems.

### Docker

```bash
docker pull ghcr.io/chimastan/historoi:latest
```

If the pull returns `denied`, Docker may be using stale GHCR credentials.

Run

```bash
docker logout ghcr.io
```

and retry the pull.

### Singularity / Apptainer

```bash
singularity pull historoi_latest.sif docker://ghcr.io/chimastan/historoi:latest
```

# Running inference

## Local execution

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results
```

## Docker

```bash
docker run --rm -it \
  -v /path/to/wsis:/data/wsis:ro \
  -v /path/to/results:/data/results \
  ghcr.io/chimastan/historoi:latest \
  python inference.py \
      --wsis "/data/wsis/*.svs" \
      --output_dir /data/results \
      --device auto
```

For GPU inference:

```bash
docker run --rm -it --gpus all \
  -v /path/to/wsis:/data/wsis:ro \
  -v /path/to/results:/data/results \
  ghcr.io/chimastan/historoi:latest \
  python inference.py \
      --wsis "/data/wsis/*.svs" \
      --output_dir /data/results \
      --device auto
```

### Docker shared memory

If PyTorch reports DataLoader worker bus errors, rerun the container with

```bash
--shm-size=2g
```

or larger, or reduce `--workers` and `--batch_size`.


## Singularity / Apptainer

```bash
singularity exec \
    -B /path/to/wsis:/data/wsis \
    -B /path/to/results:/data/results \
    historoi_latest.sif \
    bash -lc '
        cd /app/historoi

        python inference.py \
            --wsis "/data/wsis/*.svs" \
            --output_dir /data/results \
            --device auto
    '
```

For GPU inference:

```bash
singularity exec --nv ...
```

Apptainer users can replace `singularity` with `apptainer`.

# Command-line arguments

## Required arguments

* **`--wsis`**
  Input WSI(s). Accepts a single WSI file, a directory containing WSIs, or a wildcard pattern (for example, `"/path/to/wsis/*.svs"`).

* **`--output_dir`**
  Directory where inference outputs are written. Defaults to `./results`.

---

## Inference options

* **`--model6`**
  Path to the HistoROI model weights. Defaults to the bundled pretrained model. Specify this argument only to use an alternative checkpoint.

* **`--device`**
  Device used for inference. Accepted values are `auto`, `cpu`, `cuda`, and `cuda:<gpu_id>`. Defaults to `auto`, which uses CUDA when available and otherwise falls back to CPU.

* **`--batch_size`**
  Batch size used during inference. Defaults to `256`.

* **`--workers`**
  Number of PyTorch `DataLoader` worker processes. Defaults to `4`.

* **`--stride`**
  Patch stride at 10× magnification. Since HistoROI classifies a 256 × 256 pixel field of view at 10× magnification, a stride of `256` produces non-overlapping patches, whereas the default stride of `128` produces 50% overlap.

* **`--magni_0`**
  Level-0 objective magnification. If omitted, it is obtained from the WSI metadata.

* **`--level_10x`**
  OpenSlide pyramid level corresponding to approximately 10× magnification. If omitted, HistoROI automatically estimates the appropriate pyramid level from the WSI metadata.

* **`--use_level_0`**
  Read patches directly from the highest-resolution (level 0) WSI pyramid level instead of automatically selecting an approximately 10× pyramid level. The extracted level-0 patches are resized before being passed to the model. This option is primarily useful when WSI pyramid metadata is unavailable or unreliable.

* **`--vis`**
  Export the legacy QuPath-compatible point GeoJSON visualisation.

* **`--skip_if_present`**
  Skip inference for a WSI if the corresponding prediction CSV already exists.

---

## Mask-constrained inference

* **`--mask_path`**
  Path to a mask image, a directory containing masks, or a wildcard pattern matching mask files corresponding to the input WSI(s). When omitted, HistoROI uses its built-in intensity-based foreground filter.

* **`--mask_labels`**
  Comma-separated integer mask labels to include (for example, `1` or `1,2,3`). Defaults to all non-zero mask labels.

* **`--require_mask`**
  Require every input WSI to have a matching mask. This option is only valid when `--mask_path` is specified.

---

## Prediction-derived exports

* **`--export_pred_polygons`**
  Export HistoROI predictions as rectangular patch-footprint polygon GeoJSON annotations in level-0 WSI coordinates.

* **`--export_pred_mask`**
  Export HistoROI predictions as a downsampled prediction mask PNG. Mask pixel values correspond to the stable HistoROI class IDs.

* **`--mask_ds`**
  Downsampling factor applied when generating prediction masks. A value of `1` produces a full-resolution level-0 mask, while larger values produce proportionally smaller masks. Defaults to `16`.

* **`--overlap_policy`**
  Policy used when prediction patch footprints overlap during prediction-derived output export. This option is only used when `--export_pred_polygons` and/or `--export_pred_mask` is specified.

  Available policies are:

  * **`error`** — stop export if overlapping patch footprints are detected.
  * **`allow_raw`** — allow export of raw overlapping prediction polygons. This policy applies only to `--export_pred_polygons`.
  * **`retain_first`** — assign overlapping prediction mask pixels to the first prediction encountered.
  * **`retain_last`** — assign overlapping prediction mask pixels to the last prediction encountered.
  * **`highest_prob`** — assign overlapping prediction mask pixels to the prediction having the highest probability for its predicted class.

  The `retain_first`, `retain_last`, and `highest_prob` policies apply only to `--export_pred_mask`.

# Examples

## Basic inference

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results
```

## Generate the legacy QuPath visualisation

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results \
    --vis
```

## Mask-constrained inference

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results \
    --mask_path /path/to/masks \
    --mask_labels 1
```

## Export prediction polygons

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results \
    --export_pred_polygons
```

## Export prediction masks

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results \
    --export_pred_mask \
    --overlap_policy highest_prob
```

## Export prediction polygons and masks

```bash
python inference.py \
    --wsis "/path/to/wsis/*.svs" \
    --output_dir results \
    --export_pred_polygons \
    --export_pred_mask \
    --overlap_policy highest_prob
```

# Outputs

## Output directory structure

```text
results/
├── annotations/
│   └── *_historoi_pred_polygons.geojson
├── csvs/
│   ├── *.csv
│   └── historoi_prediction_summary.csv
├── logs/
│   ├── inference.log
│   ├── inference_errors.csv
│   └── run_config.json
├── masks/
│   └── *_historoi_pred_mask.png
└── visualizations/
    └── *.geojson
```

## Prediction CSV

One prediction CSV is generated for each processed WSI.

Each row corresponds to one analysed image patch.

| Column     | Description                                         |
| ---------- | --------------------------------------------------- |
| `dim1`     | Patch centre X-coordinate (level-0 WSI coordinates) |
| `dim2`     | Patch centre Y-coordinate (level-0 WSI coordinates) |
| `class_id` | Stable HistoROI class ID (1–6)                      |
| `preds`    | Predicted tissue class                              |
| `prob_*`   | Predicted probability for each tissue class         |

## Prediction summary CSV

A single file

```text
csvs/historoi_prediction_summary.csv
```

summarises every processed WSI.

Each row contains:

* WSI name
* total predicted patches
* number of predictions belonging to each tissue class
* percentage of predictions belonging to each tissue class

This provides a convenient overview of dominant and rare tissue classes across large cohorts.

## Legacy visualisation GeoJSON

When `--vis` is specified, a QuPath-compatible point-based GeoJSON visualisation is generated.

This output is retained primarily for compatibility with the original HistoROI workflow.

## Prediction polygon GeoJSON

When `--export_pred_polygons` is specified, rectangular patch-footprint polygons are exported in level-0 WSI coordinates.

Unlike the legacy point-based visualisation, these polygons represent the full prediction patch footprint and are intended for visualisation, downstream analysis and interoperability with other packages such as PySlyde.

## Prediction mask

When `--export_pred_mask` is specified, a downsampled prediction mask PNG is generated.

Pixel values correspond to stable HistoROI class IDs.

| Pixel value | Meaning                      |
| ----------- | ---------------------------- |
| 0           | No prediction / unclassified |
| 1–6         | HistoROI class IDs           |

When prediction patches overlap, mask generation follows the selected `--overlap_policy`.

# Citation

This repository builds upon the original HistoROI work.

If you use HistoROI or this inference pipeline, please cite:

```bibtex
@article{patil2023efficient,
  title={Efficient quality control of whole slide pathology images with human-in-the-loop training},
  author={Patil, Abhijeet and Diwakar, Harsh and Sawant, Jay and Kurian, Nikhil Cherian and Yadav, Subhash and Rane, Swapnil and Bameta, Tripti and Sethi, Amit},
  journal={Journal of Pathology Informatics},
  pages={100306},
  year={2023},
  publisher={Elsevier}
}
