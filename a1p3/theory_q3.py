import numpy as np

top, left, bottom, right = 11, 6, 10, 7
x, y = 6.5, 10.9

wx = (x - left) / (right - left)      # 0 at left, 1 at right
wy = (y - bottom) / (top - bottom)    # 0 at bottom, 1 at top

tl, tr, bl, br = 233, 212, 69, 234

interp = ((1 - wx) * (1 - wy) * bl
        + wx       * (1 - wy) * br
        + (1 - wx) * wy       * tl
        + wx       * wy       * tr)

print(round(interp))