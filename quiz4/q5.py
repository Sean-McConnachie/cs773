"""
Calculate and return the Root Mean Squared Error (RMSE) in 2D between
forward-projected 2D points and the actual reference 2D points (Tsai
calibration error).

RMSE = sqrt( (1/n) * sum_{i=1}^{n} [ (u_i - u~_i)^2 + (v_i - v~_i)^2 ] )

where (u_i, v_i) are the actual coordinates, (u~_i, v~_i) are the
forward-projected (estimated) coordinates, and n is the number of points.

Args:
    estimate_coordinates: forward-projected 2D points (u~, v~).
    actual_coordinates:   actual 2D reference points (u, v).

Returns:
    RMSE in 2D. Do not round.

Note: Numpy and math may be assumed available.
"""

import numpy as np
import math


def calculate_calib_error_2D(estimate_coordinates, actual_coordinates):
    # Forward-projected (u~, v~) vs measured (u, v); RMSE over all n points:
    #   RMSE = sqrt( (1/n) * sum [ (u - u~)^2 + (v - v~)^2 ] )
    estimate = np.asarray(estimate_coordinates, dtype=float)
    actual = np.asarray(actual_coordinates, dtype=float)

    squared_error = np.sum((actual - estimate) ** 2, axis=1)
    return math.sqrt(np.mean(squared_error))


if __name__ == "__main__":
    # calculate error for H3 camera
    left_corner_list = np.genfromtxt("H3_Cube_Left.csv", delimiter=",", skip_header=1, usecols=(3, 4))
    right_corner_list = np.genfromtxt("H3_Cube_Right.csv", delimiter=",", skip_header=1, usecols=(3, 4))
    left_estimate_coordinate = np.load('./H3_left_projected_2D_points.npy', allow_pickle=True)
    right_estimate_coordinate = np.load('./H3_right_projected_2D_points.npy', allow_pickle=True)

    H3_left_calib_error = calculate_calib_error_2D(left_estimate_coordinate, left_corner_list)
    H3_right_calib_error = calculate_calib_error_2D(right_estimate_coordinate, right_corner_list)

    print(f"Tsai Calibration Error in 2D (H3 Left): {round(H3_left_calib_error, 3)}")
    print(f"Tsai Calibration Error in 2D (H3 Right): {round(H3_right_calib_error, 3)}")