"""
Calculate and return the back-projection error in 3D as the Root Mean
Squared Error (RMSE) between back-projected 3D points and the actual
reference 3D points (both in the world reference frame).

RMSE = sqrt( (1/n) * sum_{i=1}^{n} [ (X_i - X~_i)^2
                                   + (Y_i - Y~_i)^2
                                   + (Z_i - Z~_i)^2 ] )

where (X_i, Y_i, Z_i) are the actual 3D coordinates, (X~_i, Y~_i, Z~_i) are
the back-projected 3D coordinates, and n is the number of points.

Args:
    estimate_coordinates: back-projected 3D points (X~, Y~, Z~) in the WRF.
    actual_coordinates:   actual 3D reference points (X, Y, Z) in the WRF.

Returns:
    RMSE in 3D. Do not round.

Note: Numpy and math may be assumed available.
"""

import numpy as np
import math


def calculate_back_projection_error(estimate_coordinates, actual_coordinates):
    # Back-projected (X~, Y~, Z~) vs true WRF (X, Y, Z); RMSE over all n points:
    #   RMSE = sqrt( (1/n) * sum [ (X-X~)^2 + (Y-Y~)^2 + (Z-Z~)^2 ] )
    estimate = np.asarray(estimate_coordinates, dtype=float)
    actual = np.asarray(actual_coordinates, dtype=float)

    squared_error = np.sum((actual - estimate) ** 2, axis=1)
    return math.sqrt(np.mean(squared_error))


if __name__ == "__main__":
    # Test for H3 left and right camera back projection error
    left_estimate_coordinates = np.load("./H3_left_projected_3D_points.npy")
    right_estimate_coordinates = np.load("./H3_right_projected_3D_points.npy")
    left_actual_coordinates = np.genfromtxt("H3_Cube_Left.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2))
    right_actual_coordinates = np.genfromtxt("H3_Cube_Right.csv", delimiter=",", skip_header=1, usecols=(0, 1, 2))

    H3_left_error = calculate_back_projection_error(left_estimate_coordinates, left_actual_coordinates)
    H3_right_error = calculate_back_projection_error(right_estimate_coordinates, right_actual_coordinates)

    print(f"Back Projection Error (H3 Left): {round(H3_left_error, 3)}")
    print(f"Back Projection Error (H3 Right): {round(H3_right_error, 3)}")

