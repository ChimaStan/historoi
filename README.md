# historoi
This repository is a fork of the original [HistoROI repository](https://github.com/abhijeetptl5/historoi), adapted as an inference-focused pipeline with improved output organisation, logging, containerised installation support, and optional mask-constrained inference.

HistoROI is a six-class classification model with the following classes:
1. Epithelial region
2. stromal region
3. Adipose / Scattered stroma, etc
4. Artefacts
5. Miscelleneous
6. Lymphocyte dense region

## Requirements

### Local installation

* Conda or Mamba.
* Python 3.10, installed through the provided `environment.yml`.
* Sufficient RAM and disk space for WSI inference outputs.
* OpenSlide support, installed through the provided conda environment.

### Containerised installation

* Docker, Singularity, or Apptainer.
* Internet access during local image build, unless pulling a prebuilt image.
* Sufficient disk space for the container image and conda environment.

### GPU inference

GPU inference is currently supported for NVIDIA CUDA GPUs only.

CUDA GPU inference requires:

* An NVIDIA GPU.
* Compatible NVIDIA drivers on the host system.
* For Docker: NVIDIA Container Toolkit and running containers with `--gpus all`.
* For Singularity/Apptainer: running containers with `--nv`.

CPU inference is supported when CUDA is unavailable or when --device cpu is used.

## Installation

### Local installation

Clone the repository and create the conda environment:

```bash
git clone https://github.com/ChimaStan/historoi.git
cd historoi

conda env create -f environment.yml
conda activate historoi
```

### Containerised installation (recommended)

Containerised installation is recommended for reproducible inference, especially on HPC systems.

#### Docker

Pull the Docker image (recommended):

```bash
docker pull ghcr.io/chimastan/historoi:latest
```

Alternatively, build one locally:
```bash
git clone https://github.com/ChimaStan/historoi.git
cd historoi

docker build -t historoi .
```

#### Singularity / Apptainer

Pull the image from the Docker registry and convert it to a Singularity/Apptainer image:

```bash
singularity pull historoi.sif docker://ghcr.io/chimastan/historoi:latest
```

## How to run

### Basic inference

Run inference locally with:

```bash
python inference.py --wsis "/path/to/wsis/*.svs" --output_dir results
```

To generate GeoJSON visualisations as well as CSV outputs, use:

```bash
python inference.py --wsis "/path/to/wsis/*.svs" --output_dir results --vis
```

### Containerised inference

#### Docker

Run inference with Docker:

```bash
docker run --rm -it \
  -v /path/to/wsis:/data/wsis:ro \
  -v /path/to/results:/data/results \
  historoi \
  python inference.py \
    --wsis "/data/wsis/*.svs" \
    --output_dir /data/results \
    --device auto
```

For NVIDIA GPU support, run the container with `--gpus all`:

```bash
docker run --rm -it --gpus all \
  -v /path/to/wsis:/data/wsis:ro \
  -v /path/to/results:/data/results \
  historoi \
  python inference.py \
    --wsis "/data/wsis/*.svs" \
    --output_dir /data/results \
    --device auto
```

If using a pulled image rather than a locally built one, replace `historoi` with:

```text
ghcr.io/chimastan/historoi:latest
```

#### Singularity / Apptainer

Run inference with Singularity:

```bash
singularity exec \
  -B /path/to/wsis:/data/wsis \
  -B /path/to/results:/data/results \
  historoi.sif \
  python inference.py \
    --wsis "/data/wsis/*.svs" \
    --output_dir /data/results \
    --device auto
```

For NVIDIA GPU support, add `--nv`:

```bash
singularity exec --nv \
  -B /path/to/wsis:/data/wsis \
  -B /path/to/results:/data/results \
  historoi.sif \
  python inference.py \
    --wsis "/data/wsis/*.svs" \
    --output_dir /data/results \
    --device auto
```

To run with Apptainer, use the keyword `apptainer` in place of `singularity`.

**Note**: When running with Docker, PyTorch `DataLoader` workers may require increased container shared memory. If inference fails with a DataLoader worker bus error, rerun the Docker container with `--shm-size=2g` or higher (e.g., `docker run --rm -it --shm-size=2g`), or reduce `--workers` and `--batch_size`. This option is specific to Docker and is usually not needed for Singularity/Apptainer workflows on HPC systems, where memory is typically managed by the job scheduler.

### Input arguments
The main arguments are:

*  `--wsis`: Path of WSI (/dir1/dir2/dir3/wsi.svs) OR Path of directory containing WSIs (/dir1/dir2/dir3/) OR Paths with wildcards (/dir1/dir2/dir3/*.svs).
*  `--output_dir`: Directory to save inference outputs, visualisations, and logs.
*  `--stride`: Stride at 10x in X and Y direction. Since HistoROI classifies a 256×256 pixel field of view at 10×, `stride=256` gives no overlap and the default `stride=128` gives 50% overlap.
*  `--batch_size`: Batch size used during inference.
*  `--workers`: Number of worker processes used by the PyTorch `DataLoader`.
* `--device`: Device to use for inference. Accepted values are `auto`, `cpu`, `cuda`, or `cuda:<gpu_id>`. If `auto`, CUDA is used when available; otherwise, CPU is used.
*  `--magni_0`: Magnification at level 0. If provided through arguments, provided value is used otherwise fetched from WSI properties.
*  `--use_level_0`: If true, patches from level 0 are extracted, resized and given as input to model. Magnification of other levels are not used.
*  `--vis`: If true, GeoJSON file compatible to QuPath is generated.
*  `--level_10x`: Level corrosponding to 10x magnification. If provided, patches from given level are extracted without reading WSI properties.
*  `--model6`: Weights for HistoROI model.
* `--skip_if_present`: Skip inference for a WSI if the corresponding CSV output already exists.

### Optional mask-constrained inference

This fork also supports restricting inference to user-provided mask regions.

*  `--mask_path`: Optional path to a mask file, a directory containing masks, or a wildcard pattern for mask files corresponding to the input WSI(s). If omitted, HistoROI uses its built-in intensity-based foreground filter.
*  `--mask_labels`: Optional comma-separated integer mask labels to include, for example `1` or `1,2,3`. Only used when `--mask_path` is provided. If omitted with `--mask_path` present, all non-zero mask values are used.
*  `--require_mask`: Require a matching mask for every input WSI. Only valid when `--mask_path` is provided.

Example using mask-constrained inference:

```bash
python inference.py \
  --wsis "/path/to/wsis/*.svs" \
  --output_dir results \
  --mask_path /path/to/masks \
  --mask_labels 1 \
  --vis
```

For containerised inference with masks, mount the mask directory as well:

```bash
docker run --rm -it --gpus all \
  -v /path/to/wsis:/data/wsis:ro \
  -v /path/to/masks:/data/masks:ro \
  -v /path/to/results:/data/results \
  historoi \
  python inference.py \
    --wsis "/data/wsis/*.svs" \
    --output_dir /data/results \
    --mask_path /data/masks \
    --mask_labels 1 \
    --device auto \
    --vis
```

## Output format

Inference results can be exported as CSV and, optionally, GeoJSON.

Coordinates in both outputs correspond to **patch centres** in the level-0 (highest-resolution) coordinate system of the WSI.

HistoROI classifies patches corresponding to a 256×256 pixel field of view at 10× magnification.

### CSV

| Column | Description |
|----------|----------|
| dim1 | X-coordinate of patch centre in level-0 WSI coordinates |
| dim2 | Y-coordinate of patch centre in level-0 WSI coordinates |
| preds | Predicted tissue class |

### GeoJSON

The GeoJSON output contains the same coordinates and predictions as the CSV output, grouped by predicted tissue class and stored as `MultiPoint` annotations compatible with QuPath.

## Output structure

By default, outputs are written to:

```text
results/
├── csvs/
├── visualizations/
└── logs/
```

## Citation

This fork builds on the original HistoROI work. If you use the model or inference outputs, please cite:

```bibtex
@article{patil2023efficient,
  title={Efficient quality control of whole slide pathology images with human-in-the-loop training},
  author={Patil, Abhijeet and Diwakar, Harsh and Sawant, Jay and Kurian, Nikhil Cherian and Yadav, Subhash and Rane, Swapnil and Bameta, Tripti and Sethi, Amit},
  journal={Journal of Pathology Informatics},
  pages={100306},
  year={2023},
  publisher={Elsevier}
}
