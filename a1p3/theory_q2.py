import numpy as np

S = {(47, 2025), (51, 2233), (60, 2442), (63, 2653), (67, 2864), (72, 3073), (77, 3282), (82, 3494), (90, 3705), (91, 3915)}

A = (67, 2864)
B = (82, 3494)

# y = mx + c
m = (B[1] - A[1]) / (B[0] - A[0])
c = A[1] - m * A[0]

THRESHOLD = 3.0
inliers = 0
for x, y in S:
    d = np.abs(m*x - y + c) / round(np.sqrt(m**2 + 1))
    print(d)
    if round(d) <= 3.0:
        inliers += 1

print(inliers)