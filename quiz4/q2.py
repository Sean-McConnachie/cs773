"""
Calculate the optical centre C of the camera in the world reference frame (WRF).

Given a calibrated camera with the following focal length and extrinsic parameters:

    R = [[ 1,  0,  0],
         [ 0,  0, -1],
         [ 0,  1,  0]]

    t = [-9, 7, -8]

    f = 4mm

Returns:
    C: 3x1 optical centre of the camera in the WRF.
"""

import numpy as np

# Extrinsic parameters map WRF -> CRF via x_CRF = R @ X_WRF + t.
# The optical centre is the CRF origin (0,0,0). Inverting the transform gives
# its WRF position:  C = -R^T @ t   (R is orthonormal, so R^-1 = R^T).
# The focal length f plays no role in locating the optical centre.
R = np.array([[1, 0,  0],
              [0, 0, -1],
              [0, 1,  0]])
t = np.array([-9, 7, -8])

C = -R.T @ t
print(f"C = {C}")
