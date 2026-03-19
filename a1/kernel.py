import numpy as np


def gaussian():
    return np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ]) * 1/16

def sobel_x():
    return np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1],
    ]) * 1/8


def sobel_y():
    return np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1],
    ]) * 1/8

def apply_kernel(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # assume border pad
    k = len(kernel)
    hk = k//2
    arr_pad = np.pad(arr, pad_width=hk, mode="edge").astype(np.float64)
    ret = np.zeros_like(arr).astype(np.float64)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            window = arr_pad[y:y+k, x:x+k]
            ret[y,x] = (window * kernel).sum()
    return ret

def round_nearest(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 0, np.floor(arr + 0.5), np.ceil(arr - 0.5)).astype(np.int64)


inp_arr = np.array([
    [141,56,124,55,225],
    [142,209,36,253,94],
    [162,79,146,156,241],
    [163,76,247,187,37],
    [236,51,4,24,169],
])

inp_arr_gaus = round_nearest(apply_kernel(inp_arr, gaussian()))
print(f"Gaussian:\n{inp_arr_gaus}")

inp_arr_sobelx = round_nearest(apply_kernel(inp_arr_gaus, sobel_x()))
print(f"Sobel X:\n{inp_arr_sobelx}")

inp_arr_sobely = round_nearest(apply_kernel(inp_arr_gaus, sobel_y()))
print(f"Sobel Y:\n{inp_arr_sobely}")

inp_arr_sobel_xx = inp_arr_sobelx*inp_arr_sobelx
print(f"Sobel X^2:\n{inp_arr_sobel_xx}")

inp_arr_sobel_yy = inp_arr_sobely*inp_arr_sobely
print(f"Sobel Y^2:\n{inp_arr_sobel_yy}")

inp_arr_sobel_xy = inp_arr_sobelx*inp_arr_sobely
print(f"Sobel XY:\n{inp_arr_sobel_xy}")

inp_arr_gsobelx = round_nearest(apply_kernel(inp_arr_sobel_xx, gaussian()))
print(f"Gaussian Sobel X^2:\n{inp_arr_gsobelx}")