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


left_image = np.array(Image.open("a1/left_image.png").convert('L'))
right_image = np.array(Image.open("a1/right_image.png").convert('L'))

image_width = 210
image_height = 200
left_image = gaussian_filtering(left_image, image_width, image_height)
right_image = gaussian_filtering(right_image, image_width, image_height)

# save image
Image.fromarray(left_image.astype(np.uint8)).save("a1/left_image_gaussian.png")
Image.fromarray(right_image.astype(np.uint8)).save("a1/right_image_gaussian.png")
