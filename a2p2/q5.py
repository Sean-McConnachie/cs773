import numpy as np
import csv
import json
from data.read_calibration_data import read_calibration_data


np.set_printoptions(suppress=True, precision=6)


def get_dx_dy(camera_type):
    if camera_type == "H3":
        dx = 0.00155
        dy = 0.00155
    elif camera_type == "W3":
        dx = 0.001691
        dy = 0.001663
    else:
        raise ValueError()
    return dx, dy


def upward_c1_xp_yp(sx, dx, cx, dy, cy, u, v):
    xp = sx * dx * (u - cx)
    yp = dy * (v - cy)
    return xp, yp


def compute_xp_yp(corner_list, callback):
    """
    Return X[0], Y[1], Z[2], u'[3], v'[4], xp[5], yp[6]
    """
    new_list = []
    for row in corner_list:
        row = list(row)
        row.extend(callback(row[3], row[4]))
        new_list.append(row)
    return np.array(new_list)


def construct_q(corner_list, f):
    last_col = np.zeros((corner_list.shape[0], 1))
    last_col[:, :] = -f
    q = corner_list[:, 5:7]
    q = np.hstack((q, last_col))
    return q


def lift_to_wrf(q, R, t):
    # (R|t)^-1 applied to each image point: M_i^W = R^T (q - t).
    # R is orthonormal so the inverse rotation is exactly its transpose.
    return (R.T @ (q - t).T).T


def compute_opt_c(R, t):
    # (R|t)^-1 applied to the CRF origin O_c = (0,0,0): O_c^W = -R^T t.
    return -R.T @ t


def intersect_plane(Mi, Oc, axis):
    ai = Mi[:, axis]
    ac = Oc[axis]
    t = -ac / (ai - ac)
    return Oc + t[:, None] * (Mi - Oc)


def back_projection(corner_list, tx, ty, tz, f, sx, R, camera_type, image_width, image_height):
    cx = image_width / 2
    cy = image_height / 2
    t = np.array([tx, ty, tz])

    # Step 0
    dx, dy = get_dx_dy(camera_type)

    # Step 1 - pixel (u, v) -> metric image coords (xp, yp)
    corner_list = compute_xp_yp(corner_list, lambda u, v: upward_c1_xp_yp(sx, dx, cx, dy, cy, u, v))

    # Step 2 - extend to CRF virtual image plane q = (xp, yp, -f)
    q = construct_q(corner_list, f)

    # Step 3 - lift the image point into the WRF (one point per ray)
    Mi = lift_to_wrf(q, R, t)

    # Step 4 - optical centre in WRF (the shared ray origin)
    opt_c = compute_opt_c(R, t)

    # Steps 5-7 - ray through (opt_c, Mi), intersect with the face each point
    # lies on: cube left face is X_w = 0, bottom face is Z_w = 0.
    Mw = np.empty((corner_list.shape[0], 3))
    on_x0 = corner_list[:, 0] == 0
    Mw[on_x0] = intersect_plane(Mi[on_x0], opt_c, axis=0)
    Mw[~on_x0] = intersect_plane(Mi[~on_x0], opt_c, axis=2)

    return Mw, opt_c




if __name__ == "__main__":
    # Back-projection on H3 camera
    left_corner_list = np.genfromtxt("data/H3_Cube_Left_v2.csv", delimiter=",", skip_header=1)
    right_corner_list = np.genfromtxt("data/H3_Cube_Right_v2.csv", delimiter=",", skip_header=1)

    image_width, image_height, sx, tx, ty, tz, f, R = read_calibration_data("H3_left")
    left_projected_3D_coordinates, left_optical_centre_WRF = back_projection(left_corner_list, tx, ty, tz, f, sx, R, "H3", image_width, image_height)
    image_width, image_height, sx, tx, ty, tz, f, R = read_calibration_data("H3_right")
    right_projected_3D_coordinates, right_optical_centre_WRF = back_projection(right_corner_list, tx, ty, tz, f, sx, R, "H3", image_width, image_height)

    expected_left = np.load("data/H3_left_projected_3D_points.npy")
    expected_right = np.load("data/H3_right_projected_3D_points.npy")
    print("left max abs error vs expected:", np.abs(left_projected_3D_coordinates - expected_left).max())
    print("right max abs error vs expected:", np.abs(right_projected_3D_coordinates - expected_right).max())

