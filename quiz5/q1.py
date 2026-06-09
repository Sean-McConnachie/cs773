"""
Compute the first-order radial distortion parameter κ₁ (barrel distortion).

Setup:
    - Image centre pixel (x, y) = (960, 540).
    - Square pixel width = 0.009000000000000001 mm.
    - Distorted pixel (x_d, y_d) = (1561, 415).
    - Constraint: undistorted point (x_u, y_u) differs from distorted by exactly 1 px:
        sqrt((x_u - x_d)^2 + (y_u - y_d)^2) = 1 px

    κ₁ is positive. Round to 2 significant figures.

Returns:
    κ₁ in units of px^-2.
"""

import math


def round_sig(x, sig=2):
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def compute_kappa_one():
    # Image centre (principal point) and the distorted pixel.
    cx, cy = 960, 540
    xd, yd = 1561, 415

    # Radial distance of the distorted point from the centre, in pixels.
    rd = math.hypot(xd - cx, yd - cy)

    # Inverse model: r_u = r_d (1 + κ₁ r_d^2), so the displacement is
    #   Δ = r_u - r_d = κ₁ r_d^3.  With Δ = 1 px:  κ₁ = Δ / r_d^3.
    # Answer is requested in px^-2, so the metric pixel width is not needed.
    delta = 1.0
    kappa_1 = delta / rd ** 3
    return kappa_1


if __name__ == "__main__":
    kappa_1 = compute_kappa_one()
    print(f"kappa_1 = {round_sig(kappa_1, 2):.10f} px^-2")
