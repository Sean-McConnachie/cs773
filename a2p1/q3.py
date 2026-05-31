


import numpy as np
from PIL import Image


def sad_cost(block_l, block_r):
    return np.abs(block_l - block_r).sum()


def reflect_make_border(im, border_size):
    return np.pad(im, pad_width=border_size, mode='symmetric')


def select_block(im, y, x, offset):
    return im[y-offset:y+offset+1, x-offset:x+offset+1]


def perform_block_matching(im_l, im_r, block_size, max_disparity):
    disp_map = np.zeros_like(im_l).astype(np.int32)

    offset = block_size // 2
    im_l = reflect_make_border(im_l, block_size // 2 + max_disparity+1)
    im_r = reflect_make_border(im_r, block_size // 2 + max_disparity+1)
    
    for y in range(disp_map.shape[0]):
        for x in range(disp_map.shape[1]):
            im_y = y + offset + max_disparity+1
            im_x = x + offset + max_disparity+1

            block_l = select_block(im_l, im_y, im_x, offset)
            block_r = select_block(im_r, im_y, im_x, offset)

            best_d = 0
            best_s = sad_cost(block_l, block_r)
            for d in range(1, min(max_disparity, x + 1)):
                block_r = select_block(im_r, im_y, im_x-d, offset)
                score = sad_cost(block_l, block_r)
                if score < best_s:
                    best_s = score
                    best_d = d
            disp_map[y, x] = best_d
    return disp_map


if __name__ == "__main__":
    rectified_left_image = np.array([
        [117,  31,  80,  81,  99,  68,  69],
        [102,  83,  92,  46,  90,  62,  54],
        [ 73,  88,  95,  99,  38,  44,  54],
        [ 61,  32,  22,  24,  25,  28,  35],
        [ 37,  51,  59,  58,  58,  62,  71],
        [ 70,  26,  31,  34,  38,  41,  49],
        [191, 178, 192, 177, 179, 179, 175],
    ])

    rectified_right_image = np.array([
        [114, 100,  85,  44,  99,  85,  59],
        [ 82,  60,  91,  47,  95,  63,  54],
        [ 67,  50,  90,  49,  39,  44,  54],
        [ 54,  31,  21,  24,  25,  27,  36],
        [ 38,  55,  59,  59,  58,  63,  71],
        [ 63,  17,  33,  37,  37,  42,  49],
        [187, 177, 188, 177, 178, 176, 173],
    ])

    disparity_map = perform_block_matching(rectified_left_image, rectified_right_image, 3, 5)
    print(disparity_map)