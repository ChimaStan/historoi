import argparse
from glob import glob
from pathlib import Path
from utils import parse_mask_labels

DEFAULT_MODEL6_PATH = Path(__file__).resolve().parent / "weights" / "model_6.pt"

def infer_options():
    parser = argparse.ArgumentParser()
    help_str = "Path of WSI (/dir1/dir2/dir3/wsi.svs) " 
    help_str += "OR Path of directory containing WSIs (/dir1/dir2/dir3/) "
    help_str += "OR Paths with wildcards (/dir1/dir2/dir3/*.svs)"
    parser.add_argument("--wsis", help=help_str, type=str, metavar='')
    
    help_str = "Directory where inference outputs, visualizations, and logs are saved."
    parser.add_argument("--output_dir", help=help_str, type=str, default='./results', metavar='DIR')
        
    help_str = "Stride at 10x in X and Y direction. (stride=256 ==> no overlap)"
    parser.add_argument("--stride", help=help_str, type=int, default=128, metavar='')
    
    parser.add_argument("--batch_size", help="Batch size", type=int, default=256, metavar='')
    parser.add_argument("--workers", help="num_workers for data loader", type=int, default=4, metavar='')
    
    help_str = "Magnification at level 0. If provided through arguments, provided value is used otherwise fetched from WSI properties"
    parser.add_argument("--magni_0", help=help_str, type=int, default=None, metavar='')
    
    help_str = "If true, patches from level 0 are extracted, resized and given as input to model. magnification of other levels are not used."
    parser.add_argument("--use_level_0", help=help_str, action='store_true')
    
    help_str = "If true, geojson file compatible to QPath is generated"
    parser.add_argument("--vis", help=help_str, action='store_true')
    
    help_str = "Level corrosponding to 10x magnification .If provided, patches from given level are extracted without reading WSI properties."
    parser.add_argument("--level_10x", help=help_str, type=int, default=None, metavar='')
    
    help_str = "Checkpoint for six-class classification model"
    parser.add_argument("--model6", help=help_str, type=str, default=str(DEFAULT_MODEL6_PATH), metavar='')
 
    parser.add_argument(
        "--device",
        help=(
            "Device to use for inference: 'auto', 'cpu', 'cuda', or 'cuda:<gpu_id>'. "
            "If 'auto', CUDA is used when available, otherwise CPU."
        ),
        type=str,
        default="auto",
        metavar="DEVICE",
    )    

    parser.add_argument(
        "--skip_if_present",
        help=(
            "Skip inference for a WSI if the corresponding CSV output already exists "
            "in the output directory."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--mask_path",
        help=(
            "Optional path to a mask file, a directory containing masks, or a wildcard " 
            "pattern for mask files corresponding to the input WSI(s). "
            "If omitted, HistoROI uses its built-in intensity-based foreground filter."
        ),
        type=str,
        default=None,
        metavar="DIR",
    )

    parser.add_argument(
        "--mask_labels",
        help=(
            "Optional comma-separated integer mask labels to include, e.g. '1' or '1,2,3'. "
            "Only used when --mask_path is provided. If omitted with --mask_path, all non-zero "
            "mask values are used."
        ),
        type=parse_mask_labels,
        default=None,
        metavar="LABELS",
    )

    parser.add_argument(
        "--require_mask",
        help=(
            "Require a matching mask for every input WSI. "
            "Only valid when --mask_path is provided."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--export_pred_polygons",
        help=(
            "Export HistoROI predictions as patch-footprint polygon GeoJSON in "
            "level-0 WSI coordinates. If patch footprints overlap, export is "
            "controlled by --overlap_policy."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--export_pred_mask",
        help=(
            "Export HistoROI predictions as a hard class mask PNG plus metadata "
            "JSON. If patch footprints overlap, overlapping mask pixels are "
            "resolved according to --overlap_policy."
        ),
        action="store_true",
    )

    parser.add_argument(
        "--mask_ds",
        help=(
            "Downsample factor for --export_pred_mask relative to level 0. "
            "Use 1 for full level-0 mask output. Default: 16."
        ),
        type=float,
        default=16.0,
        metavar="FACTOR",
    )

    parser.add_argument(
        "--overlap_policy",
        help=(
            "Policy used when HistoROI patch footprints overlap during "
            "prediction-derived output export. This option is only used when "
            "--export_pred_polygons and/or --export_pred_mask is set. "
            "'error' stops export if overlaps are present, preventing ambiguous "
            "polygon or mask outputs. "
            "'allow_raw' allows raw overlapping polygon GeoJSON export only; this "
            "is useful for visual inspection but may be unsafe for downstream "
            "rasterisation. "
            "'retain_first' keeps the first prediction assigned to each mask pixel. "
            "'retain_last' lets later predictions overwrite earlier mask pixels. "
            "'highest_prob' assigns each overlapping mask pixel to the prediction "
            "with the highest probability for its predicted class. "
            "The retain/highest_prob policies apply only to --export_pred_mask. "
            "Default: error."
        ),
        choices=["error", "allow_raw", "retain_first", "retain_last", "highest_prob"],
        default="error",
    )

    args = parser.parse_args()

    if args.mask_labels is not None and args.mask_path is None:
        parser.error("--mask_labels requires --mask_path.")

    if args.require_mask and args.mask_path is None:
        parser.error("--require_mask requires --mask_path.")

    if args.mask_path is not None:
        mask_path = Path(args.mask_path)

        if not mask_path.exists() and not glob(args.mask_path, recursive=True):
            parser.error(
                "--mask_path must be an existing mask file, an existing directory, "
                f"or a wildcard pattern matching mask files: {args.mask_path}"
            )

    return args
