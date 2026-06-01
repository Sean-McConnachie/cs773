import json
import numpy as np


def read_calibration_data(subset_name: str):
    with open("data/calibration_parameters_H3.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    if subset_name not in data:
        raise KeyError(
            f"Subset '{subset_name}' not found. "
            f"Available subsets: {list(data.keys())}"
        )

    p = data[subset_name]

    image_width = int(p["image_width"])
    image_height = int(p["image_height"])
    sx = float(p["sx"])
    tx = float(p["tx"])
    ty = float(p["ty"])
    tz = float(p["tz"])
    f = float(p["f"])
    R = np.asarray(p["R"], dtype=np.float64)
    
    return (
        image_width,
        image_height,
        sx,
        tx, ty, tz,
        f,
        R
    )
