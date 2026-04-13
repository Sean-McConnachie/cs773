import numpy as np
from PIL import Image


def compute_NCC_component(image, corner_list, window_size):    
    hws = window_size//2
    ret = []
    for y, x in corner_list:
        if y-hws < 0 or x-hws < 0 or y+hws >= image.shape[0] or x+hws >= image.shape[1]:
            continue

        window = image[y-hws:y+hws+1, x-hws:x+hws+1]
        mu = window.mean()
        zero_mean = window - mu
        zero_mean_sq = zero_mean ** 2
        zero_mean_sq_sum = zero_mean_sq.sum()
        ncc = zero_mean / np.sqrt(zero_mean_sq_sum)
        ret.append((y, x, ncc))
    return ret


if __name__ == "__main__":
    left_image = np.array(Image.open("data/feature_matching_step1_inputs/left_image.png").convert('L'))
    right_image = np.array(Image.open("data/feature_matching_step1_inputs/right_image.png").convert('L'))

    left_corner_list = np.load("data/feature_matching_step1_inputs/step4_left_nms_corner_response.npy")
    right_corner_list = np.load("data/feature_matching_step1_inputs/step4_right_nms_corner_response.npy")

    print(left_corner_list)

    window_size = 15
    NCCs_left_component = compute_NCC_component(left_image, left_corner_list, window_size)
    NCCs_right_component = compute_NCC_component(right_image, right_corner_list, window_size)

    expected_NCCs_left_component = np.load("data/feature_matching_step1_outputs/ncc_left_component.npy", allow_pickle=True)
    expected_NCCs_right_component = np.load("data/feature_matching_step1_outputs/ncc_right_component.npy", allow_pickle=True)

    # print("expected_NCCs_left_component:")
    # print(expected_NCCs_left_component)
    # print("expected_NCCs_right_component:")
    # print(expected_NCCs_right_component)

    print(f"len(left_image): {len(left_image)}")
    print(f"len(left_corner_list): {len(left_corner_list)}")
    print(f"len(expected_NCCs_left_component): {len(expected_NCCs_left_component)}")

    # print distance between expected and actual
    print(f"dist(NCCs_left_component, expected_NCCs_left_component): {np.array([ncc for _, _, ncc in NCCs_left_component]) - np.array(expected_NCCs_left_component[:, 2])}")