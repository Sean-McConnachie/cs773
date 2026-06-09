"""
Compute the estimated scene coordinates M_W = (X_W, Y_W, Z_W)^T in mm of an
image pixel p = [u, v]^T, given its depth value Z_W (in mm) from a depth map.

A calibrated pinhole camera C is placed in a 3D scene, with the origin of the
scene coordinate system at the camera's optical centre. The scene Cartesian
axes are aligned with the image plane: the image-plane x-axis and y-axis match
the scene axes, Z_c points towards the image, x_p points horizontally
rightwards, and y_p points vertically downwards (camera-geometry convention C2,
tutorial T9 pp. 8-9).

The camera has:
    - focal length f in mm
    - optical center c = [cx, cy]^T in pixels
    - no lens distortion
    - square pixels of physical size pixel_size in mm per pixel
    - horizontal scaling factor sx = 1 (s_x)

Args:
    sx:         horizontal scaling factor (assumed 1).
    f:          focal length in mm.
    cx, cy:     optical center pixel coordinates.
    pixel_size: physical pixel size in mm per pixel.
    u, v:       pixel coordinates of point p.
    Z:          depth value Z_W of pixel p in mm.

Returns:
    list [X, Y, Z]: estimated scene coordinates in mm. Do not round.
"""


import numpy as np


def reverse_projection(sx, f, cx, cy, pixel_size, u, v, Z):
    # World origin sits at the optical centre with axes aligned to the camera,
    # so R = I, t = 0 and the camera frame IS the world frame: Z_c = Z is known.
    # We simply invert the pinhole projection.

    # Pixel -> metric image plane (mm). u matches world X (no flip), but the
    # pixel v axis points down while the world Y axis points up, so y flips:
    #   x_p = sx * pixel_size * (u - cx)
    #   y_p = pixel_size * (cy - v)
    xp = sx * pixel_size * (u - cx)
    yp = pixel_size * (cy - v)

    # Invert xp = f * X / Z and yp = f * Y / Z using the known depth Z.
    X = xp * Z / f
    Y = yp * Z / f
    return [X, Y, Z]


if __name__ == "__main__":
    sx = 1
    f = 6
    cx = 178
    cy = 100
    pixel_size = 0.003
    u = 131
    v = 56
    Z = 394

    X_coord, Y_coord, Z_coord = np.round(reverse_projection(sx, f, cx, cy, pixel_size, u, v, Z), 1)
    print(f"Pixel p scene coordinate: [{X_coord}, {Y_coord}, {Z_coord}]")

