import json
import pandas as pd
from pathlib import Path
from class_labels import CLASS_COLORS_RGB

def make_geojson(csv_path, output_dir):

    df = pd.read_csv(csv_path)

    features = []
    for label, group in df.groupby('preds'):
        coords = group.iloc[:, :2].values.tolist()
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": coords
            },
            "properties": {
                "object_type": "annotation",
                "classification": {
                    "name": label,
                    "colorRGB": CLASS_COLORS_RGB[label]
                },
                "isLocked": False
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{Path(csv_path).stem}.geojson"

    with open(out_file, "w") as f:
        json.dump(geojson, f, indent=4)
    
    return out_file
