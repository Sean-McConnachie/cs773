"""
Calculate the actual square length (l) in the scene in mm.

The camera C is used to capture an image of a scene. After careful calibration,
it was determined that the camera has:
    - focal length f = 5 mm
    - optical center p_c = [119, 161]^T in pixels
    - no distortion
    - square pixels of size 0.001 mm on the sensor

A square is visible in the image with a side length of 16 pixels, centered
directly on the optical center. The square is parallel with the camera sensor
and is a distance of 486 mm from the camera in the Z direction.

Note: round the final answer to 1 decimal place.

Returns:
    l: actual side length of the square in the scene, in mm.
"""

f = 5.0          # focal length (mm)
dx = 0.001       # pixel size (mm/px), square pixels
side_px = 16     # square side length in the image (pixels)
Z = 486.0        # distance to the square along Z (mm)

# The square is parallel to the sensor, so every point shares the same depth Z.
# Pinhole magnification: image_size / world_size = f / Z.
# First convert the imaged side from pixels to mm, then invert the scaling.
# The optical-centre position is irrelevant (the square is centred on it).
side_mm = side_px * dx          # imaged side length on the sensor (mm)
l = side_mm * Z / f             # actual side length in the scene (mm)

print(f"imaged side = {side_mm} mm")
print(f"l = {l} mm -> {round(l, 1)} mm")
