import numpy as np


def checking_points_are_collinear(rand_matches):
    ...

def ransac_algorithm(matched_corner_pairs):
    pass


if __name__ == "__main__":
    filtered_matched_corner_pairs = np.load("data/filtered_matched_corner_pairs_ransac.npy")
    
    expected_H = np.array([
        [-0.014, 0, 1],
        [0, -0.014, 0],
        [0, 0, -0.014]
    ])
    
    H = ransac_algorithm(filtered_matched_corner_pairs)
    print(H)