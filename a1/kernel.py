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

inp_arr_gsobelxx = round_nearest(apply_kernel(inp_arr_sobel_xx, gaussian()))
print(f"Gaussian Sobel X^2:\n{inp_arr_gsobelxx}")

inp_arr_gsobelyy = round_nearest(apply_kernel(inp_arr_sobel_yy, gaussian()))
print(f"Gaussian Sobel Y^2:\n{inp_arr_gsobelyy}")

inp_arr_gsobelxy = round_nearest(apply_kernel(inp_arr_sobel_xy, gaussian()))
print(f"Gaussian Sobel XY:\n{inp_arr_gsobelxy}")

def compute_cornerness_score(Ixx, Iyy, Ixy, alpha, im_width, im_height):
    ret = np.zeros_like(Ixx).astype(np.float64)
    w = 3
    hw = w//2
    Ixx = np.pad(Ixx, pad_width=hw, mode="edge")
    Iyy = np.pad(Iyy, pad_width=hw, mode="edge")
    Ixy = np.pad(Ixy, pad_width=hw, mode="edge")
    for y in range(im_height):
        for x in range(im_width):
            Ixx_w = Ixx[y:y+w, x:x+w].sum().astype(np.float64)
            Iyy_w = Iyy[y:y+w, x:x+w].sum().astype(np.float64)
            Ixy_w = Ixy[y:y+w, x:x+w].sum().astype(np.float64)
            H = np.array([[Ixx_w, Ixy_w],
                          [Ixy_w, Iyy_w]])
            det_h = Ixx_w*Iyy_w - Ixy_w**2
            trace_h = Ixx_w+Iyy_w
            ret[y,x] = det_h - alpha*trace_h**2
    return ret

cornerness = round_nearest(compute_cornerness_score(inp_arr_gsobelxx, inp_arr_gsobelyy, inp_arr_gsobelxy, alpha=0.04, im_width=5, im_height=5))
print(f"Cornerness:\n{cornerness}")

threshold = 1500000
thresholded = cornerness.copy()
thresholded[cornerness <= threshold] = 0
print(f"Thresholded:\n{thresholded}")

def nms(corner_response, image_width, image_height):
    w = 3
    hw = w//2
    cr = np.pad(corner_response, pad_width=hw, mode="edge")
    ret = np.zeros_like(corner_response)
    for y in range(image_height):
        for x in range(image_width):
            patch = cr[y:y+w, x:x+w]
            if patch.max() == corner_response[y, x]:
                ret[y, x] = corner_response[y, x]
    return ret

nms = nms(thresholded, image_width=5, image_height=5)
print(f"NMS:\n{nms}")


# Ix = np.array([[2, 1, -1],
#                [3, 0, -2],
#                [2, -1, -3]])
# Iy = np.array([[1, 2, 1],
#                [0, 0, 0],
#                [-1, -2, -1]])

Ix = np.array([[141, 323, 540],
               [141, 323, 530],
               [113, 235, 329]])
Iy = np.array([[120, 152, 107],
               [120, 152, 107],
               [131, 181, 149]])
Ixy = np.array([[-48, 117, 194],
                [-48, 117, 194],
                [-24, 112, 158]])
cornerness = round_nearest(compute_cornerness_score(Ix, Iy, Ixy, alpha=0.04, im_width=3, im_height=3))
print(f"Cornerness:\n{cornerness}")