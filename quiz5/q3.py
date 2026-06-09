"""
Compute undistorted pixel coordinates for a distorted point (Tsai model).

Setup:
    - Camera resolution: 5000 x 3750 px.
    - Sensor size: 5.0 mm x 3.75 mm.
    - First-order radial distortion: κ₁ = -1.2E-3 mm^-2.
    - Distorted pixel coordinates of point M: (5000, 3750).
    - Metric reference frame origin is at the image centre.

    Round to the nearest pixel.

Returns:
    (x, y): undistorted pixel coordinates.
"""

import math


def compute_undistorted_point():
    # Resolution, sensor size, and the distorted pixel.
    w, h = 5000, 3750
    csx, csy = 5.0, 3.75
    kappa_1_m = -1.2e-3                 # mm^-2
    xd, yd = 5000, 3750

    # Principal point at the image centre.
    cx, cy = w / 2, h / 2

    # Physical pixel size (mm/px), averaged over both axes (square here).
    pixel_size = ((csx / w) + (csy / h)) / 2

    # Convert κ₁ from mm^-2 to px^-2.
    kappa_1_p = kappa_1_m * pixel_size ** 2

    # Inverse first-order radial correction, centred on the principal point.
    rd2 = (xd - cx) ** 2 + (yd - cy) ** 2
    factor = 1 + kappa_1_p * rd2
    xu = (xd - cx) * factor + cx
    yu = (yd - cy) * factor + cy

    return round(xu), round(yu)


if __name__ == "__main__":
    x, y = compute_undistorted_point()
    print(f"Undistorted pixel: ({x}, {y})")
