"""
Calculate and return the stereo calibration error as the Root Mean Squared
Error (RMSE) in 3D, comparing the stereo-estimated 3D points against the
actual 3D reference points (all in the world reference frame).

RMSE = sqrt( (1/n) * sum_{i=1}^{n} [ (X_i - X~_i)^2
                                   + (Y_i - Y~_i)^2
                                   + (Z_i - Z~_i)^2 ] )

where (X_i, Y_i, Z_i) are the actual 3D coordinates, (X~_i, Y~_i, Z~_i) are
the stereo back-projected 3D coordinates, both in the WRF, and n is the
number of points.

Args:
    left_optical_centre:           optical centre of the left camera in the WRF.
    estimated_left_3D_coordinates:  back-projected 3D points from the left image (Step 4).
    right_optical_centre:          optical centre of the right camera in the WRF.
    estimated_right_3D_coordinates: back-projected 3D points from the right image (Step 4).
    actual_coordinates:            real 3D reference points in the WRF.

Returns:
    RMSE for the stereo calibration in 3D. Do not round.

Note: Numpy and math may be assumed available.
"""
import numpy as np
import math


def calculate_stereo_calibration_error(left_optical_centre, estimated_left_3D_coordinates,
    right_optical_centre, estimated_right_3D_coordinates, actual_coordinates):
    # Each correspondence gives a left ray and a right ray in the WRF. The ray
    # passes through the optical centre and the camera's back-projected world
    # point, so:
    #   left ray :  O_L + s * d_L,  d_L = M~_L - O_L
    #   right ray:  O_R + r * d_R,  d_R = M~_R - O_R
    # In practice the two rays are skew, so the stereo estimate M~ is the
    # midpoint of the shortest segment joining them (least-squares closest
    # points). We then take the 3D RMSE of M~ against the true WRF points.
    O_L = np.asarray(left_optical_centre, dtype=float)
    O_R = np.asarray(right_optical_centre, dtype=float)
    M_L = np.asarray(estimated_left_3D_coordinates, dtype=float)
    M_R = np.asarray(estimated_right_3D_coordinates, dtype=float)
    actual = np.asarray(actual_coordinates, dtype=float)

    # Per-point ray directions and the offset between the two ray origins.
    d_L = M_L - O_L                       # (n, 3)
    d_R = M_R - O_R                       # (n, 3)
    w0 = O_L - O_R                        # (3,)

    # 2x2 normal-equation coefficients per point (skew-line midpoint).
    a = np.sum(d_L * d_L, axis=1)         # d_L . d_L
    b = np.sum(d_L * d_R, axis=1)         # d_L . d_R
    c = np.sum(d_R * d_R, axis=1)         # d_R . d_R
    d = d_L @ w0                          # d_L . w0
    e = d_R @ w0                          # d_R . w0

    denom = a * c - b * b
    s = (b * e - c * d) / denom          # parameter along the left ray
    r = (a * e - b * d) / denom          # parameter along the right ray

    closest_L = O_L + s[:, None] * d_L
    closest_R = O_R + r[:, None] * d_R
    estimate = (closest_L + closest_R) / 2.0

    squared_error = np.sum((actual - estimate) ** 2, axis=1)
    return math.sqrt(np.mean(squared_error))


if __name__ == "__main__":
    # Test for H3 camera
    left_optical_centre = np.array([38.74682, 41.51768, 17.956748])
    right_optical_centre = np.array([42.120594, 39.899624, 17.920001])
    left_estimate_coordinates = np.load("./H3_left_projected_3D_points.npy")
    right_estimate_coordinates = np.load("./H3_right_projected_3D_points.npy")
    actual_coordinates = np.genfromtxt("H3_Cube_Left.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2))

    H3_stereo_calibration_error = calculate_stereo_calibration_error(left_optical_centre, left_estimate_coordinates, right_optical_centre, right_estimate_coordinates, actual_coordinates)

    print(f"Stereo Calibration Error for H3: {round(H3_stereo_calibration_error, 3)}")