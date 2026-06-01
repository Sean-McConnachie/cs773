import numpy as np


np.set_printoptions(suppress=True, precision=6)


def forward_projection(M, f, sx, dx, dy, cx, cy, R, t):
    """
    Project a 3D scene point M (WRF) to pixel coordinates (u, v).
    C1 convention (Z_c away from the image): f is negative.
    """
    K = np.eye(3)
    K[0] = [sx / dx, 0, cx]
    K[1] = [0, 1 / dy, cy]

    # C1 image plane is at z = -f, so the metric projection is x_p = -f X/Z.
    # The I matrix therefore takes -f (the positive focal-length magnitude).
    I = np.zeros((3, 4))
    I[0, 0] = -f
    I[1, 1] = -f
    I[2, 2] = 1

    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t

    M_homo = np.append(M, 1.0)
    proj = K @ I @ Rt @ M_homo
    proj = proj / proj[2]
    return proj[:2]


if __name__ == "__main__":
    # Camera C: optical center is the WRF origin, image axes aligned with the
    # scene axes -> R = I, t = 0. C1 convention so f = -3 mm.
    f = -3.0
    sx = 1.0
    dx = dy = 0.003
    cx, cy = 199, 163
    R = np.eye(3)
    t = np.zeros(3)

    # 3D scene point
    M = np.array([-10.0, -47.0, 402.0])

    u, v = forward_projection(M, f, sx, dx, dy, cx, cy, R, t)
    print("p =", np.array([u, v]))
    print(f"u = {u:.1f} pixels")
    print(f"v = {v:.1f} pixels")
