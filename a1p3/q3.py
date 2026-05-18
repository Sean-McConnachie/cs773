import math
import numpy as np
from PIL import Image


def transform_pt(H, y, x):
    tx, ty, tw = H @ np.array([x, y, 1.0])
    return ty / tw, tx / tw

def inverse_warping(H, l_im, r_im):
    l_im, r_im = np.array(l_im), np.array(r_im)
    height, width = 200, 420
    output = np.zeros((height, width))
    
    lh, lw = l_im.shape[:2]
    rh, rw = r_im.shape[:2]
    
    for y in range(height):
        for x in range(width):
            ry, rx = transform_pt(H, y, x)
            ry, rx = round(ry), round(rx)
            
            in_left = y < lh and x < lw
            in_right = ry >= 0 and ry < rh and rx >= 0 and rx < rw
            
            if in_left and in_right:
                output[y, x] = l_im[y, x] * 0.95 + r_im[ry, rx] * 0.05
            elif in_left:
                output[y, x] = l_im[y, x]
            elif in_right:
                output[y, x] = r_im[ry, rx]
            else:
                ... # leave as black
    return output

if __name__ == "__main__":
    left_image = np.array(Image.open("data/left_image.png").convert('L'))
    right_image = np.array(Image.open("data/right_image.png").convert('L'))

    best_transformation_matrix = np.array([[-0.014,    0.0,    1.0],
                                           [   0.0, -0.014,    0.0],
                                           [   0.0,    0.0, -0.014]])
    warped_image = inverse_warping(best_transformation_matrix, left_image, right_image)
    # save
    im = Image.fromarray(warped_image.astype(np.uint8))
    im.save("warped_image.png")