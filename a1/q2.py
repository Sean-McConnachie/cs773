import numpy as np
from PIL import Image


def gausian_kernel():
    return np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ])


def apply_kernel(im: np.ndarray, kernel: np.ndarray, im_width: int, im_height: int) -> np.ndarray:
    ksize = kernel.shape[0]
    hksize = ksize // 2
    patch = np.zeros((ksize, ksize))
    ret = np.zeros_like(im, dtype=np.float128)
    for y in range(im_height):
        for x in range(im_width):
            for ky in range(ksize):
                for kx in range(ksize):
                    patch[ky, kx] = im[max(0, min(y+ky-hksize, im_height-1)),
                                       max(0, min(x+kx-hksize, im_width-1)) ]
            ret[y, x] = (patch * kernel).sum()
    return ret


def gaussian_filtering(image, image_width, image_height):
    return apply_kernel(image, gausian_kernel(), image_width, image_height)


def sobel_x() -> np.ndarray:
    return np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1],
    ]) * 1/8


def sobel_y() -> np.ndarray:
    return np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1],
    ]) * 1/8


def compute_image_derivatives(image, image_width, image_height):
    Ix = apply_kernel(image, sobel_x(), image_width, image_height)
    Iy = apply_kernel(image, sobel_y(), image_width, image_height)
    return Ix*Ix, Iy*Iy, Ix*Iy


def compute_cornerness_score(Ixx, Iyy, Ixy, alpha, threshold, im_width, im_height): 
    cornerness = (Ixx * Iyy - Ixy * Ixy) - alpha * (Ixx + Iyy) ** 2
    ys, xs = np.where(cornerness > threshold)
    scores = cornerness[ys, xs]
    indicies = np.argsort(scores)[::-1][:1000]
    return [ (xs[i], ys[i], scores[i]) for i in indicies ]

def compute_cornerness_score_naive(Ixx, Iyy, Ixy, alpha, threshold, im_width, im_height):
    pts = []
    for y in range(im_height):
        for x in range(im_width):
            lcl_Ixx = Ixx[y, x]
            lcl_Iyy = Iyy[y, x]
            lcl_Ixy = Ixy[y, x]
            det_H = lcl_Ixx*lcl_Iyy - lcl_Ixy*lcl_Ixy  # ad-bc
            trc_H = lcl_Ixx+lcl_Iyy
            cornerness = det_H - alpha * trc_H**2
            if cornerness > threshold:
                pts.append((x, y, cornerness))
    pts = sorted(pts, key=lambda v: v[2], reverse=True)
    return pts[:1000]


l_im = np.array(Image.open("a1/left_image.png").convert('L'))
r_im = np.array(Image.open("a1/right_image.png").convert('L'))

# image_width = 210
# image_height = 200
# lg_im = gaussian_filtering(l_im, image_width, image_height)
# rg_im = gaussian_filtering(r_im, image_width, image_height)

# l_Ixx, l_Iyy, l_Ixy = compute_image_derivatives(lg_im, image_width, image_height)
# r_Ixx, r_Iyy, r_Ixy = compute_image_derivatives(rg_im, image_width, image_height)


image_width = 210
image_height = 200
l_Ixx = np.load("a1/step2_expected_outputs/step2_left_ix_square.npy")
l_Iyy = np.load("a1/step2_expected_outputs/step2_left_iy_square.npy")
l_Ixy = np.load("a1/step2_expected_outputs/step2_left_ixiy.npy")
r_Ixx = np.load("a1/step2_expected_outputs/step2_right_ix_square.npy")
r_Iyy = np.load("a1/step2_expected_outputs/step2_right_iy_square.npy")
r_Ixy = np.load("a1/step2_expected_outputs/step2_right_ixiy.npy")

l_gIxx = gaussian_filtering(l_Ixx, image_width, image_height)
l_gIyy = gaussian_filtering(l_Iyy, image_width, image_height)
l_gIxy = gaussian_filtering(l_Ixy, image_width, image_height)
r_gIxx = gaussian_filtering(r_Ixx, image_width, image_height)
r_gIyy = gaussian_filtering(r_Iyy, image_width, image_height)
r_gIxy = gaussian_filtering(r_Ixy, image_width, image_height)

cornerness_score_left = compute_cornerness_score(l_gIxx, l_gIyy, l_gIxy, alpha=0.04, threshold=10**6, im_width=image_width, im_height=image_height)
cornerness_score_right = compute_cornerness_score(r_gIxx, r_gIyy, r_gIxy, alpha=0.04, threshold=10**6, im_width=image_width, im_height=image_height)


# left_ix_square = np.load("a1/step3_expected_data/step3_blurred_left_ix_square.npy")
# left_iy_square = np.load("a1/step3_expected_data/step3_blurred_left_iy_square.npy")
# left_ixiy = np.load("a1/step3_expected_data/step3_blurred_left_ixiy.npy")
# right_ix_square = np.load("a1/step3_expected_data/step3_blurred_right_ix_square.npy")
# right_iy_square = np.load("a1/step3_expected_data/step3_blurred_right_iy_square.npy")
# right_ixiy = np.load("a1/step3_expected_data/step3_blurred_right_ixiy.npy")

alpha = 0.04
threshold = 10 ** 6
# cornerness_score_left = compute_cornerness_score(left_ix_square, left_iy_square, left_ixiy, alpha, threshold, image_width, image_height)
# cornerness_score_right = compute_cornerness_score(right_ix_square, right_iy_square, right_ixiy, alpha, threshold, image_width, image_height)

# plot
def plot_corner_pts(im, pts, save_path):
    im = Image.fromarray(im.astype(np.uint8)).convert('RGB')
    for x, y, _ in pts:
        im.putpixel((x, y), (255, 0, 0))
    im.save(save_path)

plot_corner_pts(l_im, cornerness_score_left, "a1/left_corners.png")
plot_corner_pts(r_im, cornerness_score_right, "a1/right_corners.png")

print(len(cornerness_score_left), len(cornerness_score_right))

# print(verify_student_answers(cornerness_score_left, cornerness_score_right))
