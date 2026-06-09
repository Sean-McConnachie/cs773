import math


def compute_max_kappa_one(w, h, csx, csy):
    """
    Determine the maximum allowable magnitude of the first-order radial
    distortion coefficient κ₁:max for a barrel-distorted imaging system.

    Constraint:
        Over the entire field of view, the maximum distance between any
        distorted image point and its undistorted location must not exceed
        the physical width of a single pixel.

    Args:
        w (int):   image width in pixels.
        h (int):   image height in pixels.
        csx (float): sensor physical dimension in x (mm).
        csy (float): sensor physical dimension in y (mm).

    Returns:
        float: κ₁:max in units of mm^-2.

    Notes:
        - Do not round the result.
        - The `math` library may be assumed available.
    """
    # Principal point at the image centre.
    cx, cy = w / 2, h / 2

    # Worst case: the corner farthest from the centre. With a centred
    # principal point all four corners are equidistant, e.g. (0, 0).
    rd = math.hypot(0 - cx, 0 - cy)

    # Physical pixel size (mm/px), averaged over both axes.
    pixel_size = ((csx / w) + (csy / h)) / 2

    # Max displacement Δ = 1 px = κ₁ r_d^3  ⇒  κ₁^p = 1 / r_d^3 (px^-2),
    # then convert to metric mm^-2.
    kappa_1_p = 1.0 / rd ** 3
    kappa_1_m = kappa_1_p / pixel_size ** 2
    return kappa_1_m
