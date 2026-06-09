import math


def compute_undistorted_coordinates(kappa_1, distorted_corners, image_height, image_width):
    """
    Remove first-order radial distortion via the inverse radial distortion model.

    Args:
        kappa_1 (float): first-order radial distortion coefficient.
        distorted_corners (list[tuple]): distorted corner locations (x, y).
        image_height (int): image height in pixels.
        image_width (int):  image width in pixels.

    Returns:
        list[tuple]: undistorted corner locations (x, y).

    Notes:
        - `math` is assumed already imported.
        - Round computed x_u and y_u up to integers using math.ceil().
    """
    # Principal point at the image centre.
    cx, cy = image_width / 2, image_height / 2

    undistorted_corners = []
    for xd, yd in distorted_corners:
        # Inverse first-order radial model, centred on the principal point.
        # kappa_1 is in pixel units (px^-2), so r_d stays in pixels.
        rd2 = (xd - cx) ** 2 + (yd - cy) ** 2
        factor = 1 + kappa_1 * rd2
        xu = (xd - cx) * factor + cx
        yu = (yd - cy) * factor + cy
        undistorted_corners.append((math.ceil(xu), math.ceil(yu)))

    return undistorted_corners
