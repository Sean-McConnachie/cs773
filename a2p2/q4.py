import numpy as np


np.set_printoptions(suppress=True, precision=6)


def c2_xp_yp(sx, dx, cx, dy, cy, u, v):
    # C2 convention: x_p uses (u - cx); y_p is flipped relative to the pixel
    # row, y_p = dy(cy - v).
    xp = sx * dx * (u - cx)
    yp = dy * (cy - v)
    return xp, yp


def lift_to_wrf(q, R, t):
    # (R|t)^-1 applied to an image point: M_i^W = R^T (q - t).
    # R is orthonormal so the inverse rotation is exactly its transpose.
    return R.T @ (q - t)


def compute_opt_c(R, t):
    # (R|t)^-1 applied to the CRF origin O_c = (0,0,0): O_c^W = -R^T t.
    return -R.T @ t


def intersect_plane(Mi, Oc, axis):
    # Ray P(s) = Oc + s(Mi - Oc); solve P[axis] = 0 and back-substitute.
    s = -Oc[axis] / (Mi[axis] - Oc[axis])
    return Oc + s * (Mi - Oc)


def back_projection(u, v, f, sx, dx, dy, cx, cy, R, t, plane_axis):
    # Step 1 - pixel (u, v) -> metric image coords (xp, yp)
    xp, yp = c2_xp_yp(sx, dx, cx, dy, cy, u, v)

    # Step 2 - extend to CRF virtual image plane.
    # The virtual image point sits in front of the camera at depth |f|.
    # C2 here gives f > 0, so the CRF z-coordinate is +f.
    # (In q5's Tsai/C1 convention f is stored negative, hence -f there.)
    q = np.array([xp, yp, f])

    # Step 3 - lift the image point into the WRF (one point on the ray)
    Mi = lift_to_wrf(q, R, t)

    # Step 4 - optical centre in WRF (the ray origin)
    Oc = compute_opt_c(R, t)

    # Steps 5-7 - intersect the ray with the constraint plane
    Mw = intersect_plane(Mi, Oc, plane_axis)
    return Mw


if __name__ == "__main__":
    # Camera C: square pixels of 0.002 mm, f = 3 mm, no distortion, sx = 1.
    f = 3.0
    sx = 1.0
    dx = dy = 0.002
    cx, cy = 221, 108

    # Extrinsic Rt (WRF -> CRF)
    Rt = np.array([
        [0.848,  0.0, 0.5299, -44.0],
        [0.0,    1.0, 0.0,    -42.0],
        [-0.5299, 0.0, 0.848, -51.0],
        [0.0,    0.0, 0.0,      1.0],
    ])
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    # Corner projected from the Z_w = 0 plane to pixel p = (201, 63)
    u, v = 201, 63
    plane_axis = 2  # Z_w = 0

    Mw = back_projection(u, v, f, sx, dx, dy, cx, cy, R, t, plane_axis)
    print("M_w =", Mw)
    print(f"X_w = {Mw[0]:.1f} mm")
    print(f"Y_w = {Mw[1]:.1f} mm")
    print(f"Z_w = {Mw[2]:.1f} mm")
