import numpy as np
import pandas as pd
import torch
from glob import glob
from pathlib import Path
from argparse import Namespace
from PIL import Image, UnidentifiedImageError
from openslide import OpenSlide
from torchvision import transforms


SUPPORTED_WSI_EXTENSIONS = {
    ".svs",
    ".ndpi",    
    ".tif",
    ".tiff",
    ".mrxs",
    ".scn",
    ".bif",
    ".vms",
    ".vmu",
    ".dcm",
    ".svslide",
    ".czi",
    ".avs",
}

def resolve_wsi_paths(wsis: str) -> list[Path]:
    """Resolve a WSI file, directory, or wildcard pattern into WSI paths."""
    wsi_path = Path(wsis)

    if wsi_path.is_file():
        return [wsi_path]

    if wsi_path.is_dir():
        paths = [
            path
            for path in wsi_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_EXTENSIONS
        ]
        paths = sorted(paths)

        if not paths:
            raise FileNotFoundError(
                f"No supported WSI files found in directory: {wsi_path}"
            )

        return paths

    paths = [
        Path(path)
        for path in glob(wsis, recursive=True)
        if Path(path).is_file()
    ]
    paths = sorted(paths)

    if not paths:
        raise FileNotFoundError(
            f"No WSI files found for input path or pattern: {wsis}"
        )

    return paths


def resolve_mask_paths(mask: str | None) -> list[Path] | None:
    """Resolve a mask file, directory, or wildcard pattern into mask paths."""
    if mask is None:
        return None

    path = Path(mask)

    if path.is_file():
        return [path]

    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file())

    paths = sorted(Path(p) for p in glob(mask) if Path(p).is_file())

    if not paths:
        raise FileNotFoundError(f"No mask files found for: {mask}")

    return paths



def resolve_device(device_arg: str) -> torch.device:
    """Resolve a user-provided device argument into a PyTorch device."""
    device_arg = device_arg.lower()

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda" or device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested with --device {device_arg!r}, "
                "but CUDA is not available."
            )

        device = torch.device(device_arg)

        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {device_arg}, but only {torch.cuda.device_count()} "
                "CUDA device(s) are available."
            )

        return device

    raise ValueError(
        f"Unsupported device: {device_arg!r}. "
        "Use 'auto', 'cpu', 'cuda', or 'cuda:<gpu_id>'."
    )


def ccrop(crop_size):
    transforms_ =  transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
    ])
    return transforms_


def get_metadata(wsi: OpenSlide, args: Namespace) -> tuple[int, int, int, int]:
    """
    Determine the WSI pyramid level and patch size used for HistoROI inference.

    HistoROI classifies image regions corresponding to a 256 x 256 pixel field
    of view at 10x magnification. This function converts that field of view into
    the appropriate patch size for the selected OpenSlide pyramid level.

    Parameters
    ----------
    wsi : openslide.OpenSlide
        OpenSlide whole-slide image object.

    args : argparse.Namespace
        Inference arguments. The function uses ``args.level_10x``,
        ``args.magni_0``, and ``args.use_level_0``.

    Returns
    -------
    tuple[int, int, int, int]
        Metadata required for patch extraction:

        - ``level``: OpenSlide pyramid level used for patch extraction.
        - ``ps``: Patch size in pixels at the selected OpenSlide level.
        - ``magni_0``: Estimated or user-provided magnification at level 0.
        - ``ps_level0``: Patch footprint in level-0 pixels.

    Raises
    ------
    ValueError
        If required WSI metadata is missing, malformed, or produces invalid
        patch extraction parameters.
    """
    if args.magni_0 is not None:
        magni_0 = args.magni_0
    else:
        try:
            mpp_x = float(wsi.properties["openslide.mpp-x"])
        except KeyError as exc:
            raise ValueError(
                "Missing WSI metadata: 'openslide.mpp-x'. "
                "Provide level-0 magnification manually using --magni_0."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid WSI metadata: 'openslide.mpp-x' must be numeric. "
                "Provide level-0 magnification manually using --magni_0."
            ) from exc

        if mpp_x <= 0:
            raise ValueError(
                f"Invalid WSI metadata: 'openslide.mpp-x' must be positive, got {mpp_x}."
            )

        magni_0 = round(10 / mpp_x)

    if magni_0 <= 0:
        raise ValueError(f"Level-0 magnification must be positive, got {magni_0}.")

    ps_level0 = round(256 * magni_0 / 10)

    if ps_level0 <= 0:
        raise ValueError(
            f"Computed invalid level-0 patch size: {ps_level0}."
        )

    if args.level_10x is not None:
        level = args.level_10x

        if level < 0 or level >= wsi.level_count:
            raise ValueError(
                f"--level_10x must be a valid OpenSlide level index. "
                f"Got {level}, but WSI has {wsi.level_count} levels."
            )

        ps = 256
        return level, ps, magni_0, ps_level0

    if args.use_level_0:
        level = 0
        ps = ps_level0
        return level, ps, magni_0, ps_level0

    try:
        mpp_x = float(wsi.properties["openslide.mpp-x"])
    except KeyError as exc:
        raise ValueError(
            "Missing WSI metadata: 'openslide.mpp-x'. "
            "Provide --level_10x or use --use_level_0 with --magni_0."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid WSI metadata: 'openslide.mpp-x' must be numeric. "
            "Provide --level_10x or use --use_level_0 with --magni_0."
        ) from exc

    if mpp_x <= 0:
        raise ValueError(
            f"Invalid WSI metadata: 'openslide.mpp-x' must be positive, got {mpp_x}."
        )

    ds = round(1 / mpp_x)
    level = wsi.get_best_level_for_downsample(ds + 0.1)

    try:
        level_ds = float(wsi.properties[f"openslide.level[{level}].downsample"])
    except KeyError as exc:
        raise ValueError(
            f"Missing WSI metadata: 'openslide.level[{level}].downsample'."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid WSI metadata: 'openslide.level[{level}].downsample' "
            "must be numeric."
        ) from exc

    if level_ds <= 0:
        raise ValueError(
            f"Invalid downsample for level {level}: expected positive value, got {level_ds}."
        )

    ps = round(ps_level0 / level_ds)

    if ps <= 0:
        raise ValueError(
            f"Computed invalid patch size {ps} at level {level}."
        )

    return level, ps, magni_0, ps_level0


def filtered_patches(
    wsi: OpenSlide,
    stride_level0: int,
    mask: np.ndarray | None = None,
    mask_labels: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate candidate patch centre coordinates for HistoROI inference.

    Output coordinates are:
        - patch centres, not patch origins
        - in level-0 WSI coordinates

    Parameters
    ----------
    wsi : openslide.OpenSlide
        OpenSlide WSI object.

    stride_level0 : int
        Stride at level-0 resolution

    mask : np.ndarray, optional
        Optional binary or labelled mask. If provided, candidate patch centres
        are kept only where the mask is positive or where the mask value is in
        mask_labels.

        The mask is assumed to correspond spatially to the WSI but may be at
        lower resolution.

    mask_labels : list, tuple, set, optional
        Label values to accept from a labelled mask.
        If None, all mask values > 0 are accepted.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            dim1: patch centre x-coordinate in level-0 WSI coordinates
            dim2: patch centre y-coordinate in level-0 WSI coordinates
    """

    if stride_level0 <= 0:
        raise ValueError(f"stride_level0 must be positive, got {stride_level0}.")

    w, h = wsi.dimensions

    thumb_w = max(1, w // stride_level0)
    thumb_h = max(1, h // stride_level0)

    if mask is None:
        thumb = np.array(
            wsi.get_thumbnail((thumb_w, thumb_h)).convert("L")
        )

        arr = np.logical_and(thumb < 240, thumb > 20)
    else:
        mask_arr = resize_mask(mask, (thumb_h, thumb_w))

        if mask_labels is None:
            arr = mask_arr > 0
        else:
            arr = np.isin(mask_arr, list(mask_labels))

    rows, cols = np.where(arr)

    df = pd.DataFrame({
        "dim1": stride_level0 * cols + (stride_level0 // 2),
        "dim2": stride_level0 * rows + (stride_level0 // 2),
    })

    return df


def parse_mask_labels(labels: str | None) -> list[int] | None:
    """
    Parse integer mask labels provided through ``--mask_labels``.

    Parameters
    ----------
    labels : str or None
        Comma-separated integer mask labels to include, for example ``"1"``
        or ``"1,2,3"``. If ``None``, downstream mask filtering should accept
        all non-zero mask values.

    Returns
    -------
    list[int] or None
        Parsed integer mask labels, or ``None`` when no explicit labels are
        provided.

    Raises
    ------
    ValueError
        If any provided value cannot be parsed as an integer.

    Notes
    -----
    For GrandQC-style labelled masks, the expected label convention is:

    - ``1``: Normal tissue
    - ``2``: Fold
    - ``3``: Dark spot / foreign object
    - ``4``: Pen marking
    - ``5``: Edge / air bubble
    - ``6``: Out of focus
    - ``7``: Background
    """
    if labels is None:
        return None

    try:
        return [int(v.strip()) for v in labels.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(
            f"--mask_labels must contain comma-separated integer labels, got: {labels!r}"
        ) from exc


def find_matching_mask(
    wsi_path: str | Path,
    mask_paths: list[Path] | None,
    require_mask: bool = False,
) -> Path | None:
    """
    Find a mask file corresponding to a WSI by matching filename.

    Supports masks named using the WSI stem or full WSI filename, followed by
    common separators such as ``_``, ``-``, or ``.``.

    Parameters
    ----------
    wsi_path : str or pathlib.Path
        Path to the input whole-slide image.

    mask_paths : list[pathlib.Path] or None
        Candidate mask file paths.

    require_mask : bool, default=False
        If ``True``, raise an error when no corresponding mask is found.

    Returns
    -------
    pathlib.Path or None
        Path to the matching mask file, or ``None`` if no mask is found and
        ``require_mask`` is ``False``.

    Raises
    ------
    FileNotFoundError
        If ``require_mask`` is ``True`` and no matching mask is found.

    RuntimeError
        If multiple matching mask files are found.
    """
    if mask_paths is None:
        return None

    wsi_path = Path(wsi_path)

    if len(mask_paths) == 1:
        return mask_paths[0]

    wsi_stem = wsi_path.stem
    wsi_name = wsi_path.name
    separators = ("_", "-", ".")

    candidates = []

    for path in mask_paths:
        mask_stem = path.stem
        mask_name = path.name

        stem_match = (
            mask_stem == wsi_stem
            or any(mask_stem.startswith(f"{wsi_stem}{sep}") for sep in separators)
        )

        name_match = any(
            mask_name.startswith(f"{wsi_name}{sep}") for sep in separators)

        if stem_match or name_match:
            candidates.append(path)

    candidates = sorted(candidates)

    if len(candidates) == 0:
        if require_mask:
            raise FileNotFoundError(f"No matching mask found for WSI: {wsi_path}")
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple masks found for WSI '{wsi_path}': "
            f"{[str(path) for path in candidates]}"
        )

    return candidates[0]


def load_image_mask(
        mask_path: str | Path,
        max_image_pixels: int | None = 1_000_000_000,
) -> np.ndarray:
    """
    Load an image mask from disk as a two-dimensional NumPy array.

    The mask is read without normalising or remapping pixel values. This
    preserves binary values such as ``0``/``255`` and labelled-mask values such
    as ``1``, ``2``, or ``3`` for downstream filtering.

    Parameters
    ----------
    mask_path : str or pathlib.Path
        Path to the image mask file.

    Returns
    -------
    numpy.ndarray
        Two-dimensional mask array.

    Raises
    ------
    FileNotFoundError
        If ``mask_path`` does not exist.

    ValueError
        If ``mask_path`` is not a file, cannot be read as an image, or does not
        contain a two-dimensional mask.

    Notes
    -----
    Masks should preferably be stored in a lossless format such as PNG or TIFF.
    Lossy formats such as JPEG may alter pixel values through compression and
    corrupt binary or labelled segmentation masks.
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

    if not mask_path.is_file():
        raise ValueError(f"Mask path is not a file: {mask_path}")
    
    previous_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_image_pixels    

    try:
        with Image.open(mask_path) as img:
            mask = np.array(img)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read mask as an image: {mask_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not open mask file: {mask_path}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_max_pixels
        
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask with shape (height, width), got shape {mask.shape} "
            f"for file: {mask_path}"
        )

    return mask


def resize_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary or labelled mask to a target spatial shape.

    The mask is resized using nearest-neighbour sampling to preserve discrete
    pixel values. This is suitable for binary masks and labelled segmentation
    masks, where interpolation would create invalid intermediate values.

    Parameters
    ----------
    mask : numpy.ndarray
        Two-dimensional input mask with shape ``(height, width)``.

    target_shape : tuple[int, int]
        Target spatial shape as ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        Resized mask with shape ``target_shape``.

    Raises
    ------
    ValueError
        If ``mask`` is not two-dimensional or if ``target_shape`` contains
        non-positive dimensions.
    """
    mask = np.asarray(mask)

    target_h, target_w = target_shape

    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"target_shape must contain positive dimensions, got {target_shape}."
        )

    source_h, source_w = mask.shape

    row_idx = np.linspace(0, source_h - 1, target_h).round().astype(int)
    col_idx = np.linspace(0, source_w - 1, target_w).round().astype(int)

    return mask[row_idx[:, None], col_idx]
