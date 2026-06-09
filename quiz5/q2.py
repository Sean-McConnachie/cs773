"""
Compute the maximum first-order radial distortion parameter κ₁:max (pin-cushion).

Setup:
    - Camera resolution: 4011 x 1067 px.
    - Sensor size: 6 mm x 13 mm.
    - Requirement: max distance between distorted and undistorted coordinates
      of ANY image point stays below one pixel width.
        r_d^2 = (x_d - c_x)^2 + (y_d - c_y)^2

    Hint 1: Distortion is largest at the furthest point from the image centre.
    Hint 2: Assume a negative distortion value.
    Hint 3: Round to 2 significant figures.

Returns:
    κ₁:max in units of px^-2.
"""

import math


def round_sig(x, sig=2):
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def compute_max_kappa_one():
    # Resolution and principal point (image centre).
    w, h = 4011, 1067
    cx, cy = w / 2, h / 2

    # Worst case is the corner farthest from the centre; for a centred
    # principal point every corner is equidistant, e.g. (0, 0).
    rd = math.hypot(0 - cx, 0 - cy)

    # Δ = κ₁ r_d^3 with Δ = 1 px gives the magnitude; pin-cushion ⇒ κ₁ < 0.
    # Answer requested in px^-2, so sensor size is not needed.
    delta = 1.0
    kappa_1_max = -delta / rd ** 3
    return kappa_1_max


if __name__ == "__main__":
    kappa_1_max = compute_max_kappa_one()
    print(f"kappa_1:max = {round_sig(kappa_1_max, 2):.11f} px^-2")
