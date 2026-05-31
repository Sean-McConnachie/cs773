import numpy as np
from PIL import Image


def ncc_cost(block_l, block_r):
    block_l_mean = np.mean(block_l)
    block_r_mean = np.mean(block_r)
    
    numerator = np.sum((block_l - block_l_mean) * (block_r - block_r_mean))
    denominator = np.sqrt(np.sum((block_l - block_l_mean)**2) * np.sum((block_r - block_r_mean)**2))
    
    if denominator == 0:
        return 0  # Avoid division by zero, can also return a small value or handle differently
    
    return numerator / denominator


def reflect_make_border(im, border_size):
    return np.pad(im, pad_width=border_size, mode='reflect')


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
            best_s = ncc_cost(block_l, block_r)
            for d in range(1, max_disparity):
                block_r = select_block(im_r, im_y, im_x-d, offset)
                score = ncc_cost(block_l, block_r)
                if score > best_s:
                    best_s = score
                    best_d = d
            disp_map[y, x] = best_d
    return disp_map


if __name__ == "__main__":
    rectified_left_image = np.array(Image.open("data/Djembe_left.png").convert('L'))
    rectified_right_image = np.array(Image.open("data/Djembe_right.png").convert('L'))
    disparity_map = perform_block_matching(rectified_left_image, rectified_right_image, 11, 20)

    expected_disparity_map = np.load("data/disparity_map.npy")
    n = 20
    print(disparity_map[:n, :n])
    print(expected_disparity_map[:n, :n])

    print("=" * 10)
    print(disparity_map[:n, :n] - expected_disparity_map[:n, :n])