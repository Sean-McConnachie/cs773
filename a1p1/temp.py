import numpy as np
from scipy.ndimage import convolve

def normal_round(arr):
    # Symmetric rounding (half round away from zero) to ensure negatives round correctly
    return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(int)

inp_arr = np.array([
    [141,  56, 124,  55, 225],
    [142, 209,  36, 253,  94],
    [162,  79, 146, 156, 241],
    [163,  76, 247, 187,  37],
    [236,  51,   4,  24, 169]
], dtype=float)

# 1. Kernels
gauss_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16.0

# Updated Sobel X Kernel
sobel_x_kernel = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
]) / 8.0

# Updated Sobel Y Kernel
sobel_y_kernel = np.array([
    [ 1,  2,  1],
    [ 0,  0,  0],
    [-1, -2, -1]
]) / 8.0

# 2. Gaussian Smoothing
smoothed = convolve(inp_arr, gauss_kernel, mode='nearest')
smoothed_rounded = normal_round(smoothed)

# 3. Sobel X on the smoothed image
sobel_x = convolve(smoothed_rounded.astype(float), sobel_x_kernel, mode='nearest')
sobel_x_rounded = normal_round(sobel_x)

# 4. Sobel Y on the smoothed image
sobel_y = convolve(smoothed_rounded.astype(float), sobel_y_kernel, mode='nearest')
sobel_y_rounded = normal_round(sobel_y)

print("Smoothed Image:\n", smoothed_rounded)
print("\nSobel X:\n", -sobel_x_rounded)
print("\nSobel Y:\n", -sobel_y_rounded)