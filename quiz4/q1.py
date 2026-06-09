"""
Assume that we have a camera C in a scene, where we have decided that the origin of the coordinate system should be the optical center of the camera. We have also aligned the scenes cartesian coordinate axes so that the x-axis and y-axis of the image plane match the scene axes, the Zc axis is pointing towards the image, xp is pointing rightwards and yp is pointing downwards, thus, the coordinate system convention in camera geometry is C2 (Please check tutorial slides T9 pages 8-9). After calibration, we have found our focal length to be 10 mm, the optical center to be at pixel coordinate [262, 282]T in the image, sx is equal to 1, and no distortion. It was also found that pixels were square and were 0.001 mm on the image sensor.

Consider the pixel p = [305, 320]T within the image. If a depth map revealed the 3D location of this coordinate to have ZW = 350.0 mm, what would the scene coordinate M̃W = [X̃W, ỸW, Z̃W]T be for the pixel p in mm units?

Note: round your final results to 1 decimal place.
"""

import numpy as np

# --- Calibration parameters ------------------------------------------------
f = 10.0                 # focal length (mm)
cx, cy = 262.0, 282.0    # principal point / optical centre (pixels)
sx = 1.0                 # x-pixel scale factor
dx = dy = 0.001          # pixel size (mm/px), square pixels, no distortion

# Pixel of interest and its known depth
u, v = 305.0, 320.0
Zw = 350.0               # depth from the depth map (mm)

# Because the WRF origin is placed at the optical centre and the scene axes are
# aligned with the camera axes, the extrinsic transform is identity
# (R = I, t = 0). The camera frame therefore *is* the world frame, so
# Zc = Zw = 350.0 mm and no ray/plane intersection is required -- we simply
# invert the pinhole projection.

# Step 1: pixel -> metric image-plane coordinates (mm).
# u increases rightwards and matches the world X axis, so no flip on x.
# The pixel v axis increases DOWNWARDS, but the camera/world Y axis points UP
# (right-handed frame: Zc toward the scene, xp right => Y up). That single-axis
# mismatch means y picks up a sign flip: yp = dy*(cy - v).
xp = sx * dx * (u - cx)
yp = dy * (cy - v)
print(f"xp = {xp} mm")
print(f"yp = {yp} mm")

# Step 2: invert perspective projection xp = f * X/Z, yp = f * Y/Z using the
# known depth Z = Zw.
Xw = xp * Zw / f
Yw = yp * Zw / f

Mw = np.array([Xw, Yw, Zw])
print(f"M~W (exact)   = {Mw}")
print(f"M~W (1 d.p.)  = {np.round(Mw, 1)}")
