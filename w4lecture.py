# import numpy as np

# q = np.array([10, 200, 10, 10, 200, 10, 200, 200, 200])
# p1 = np.array([12, 198, 12, 10, 202, 11, 198, 201, 199])
# p2 = np.array([8, 201, 9, 11, 198, 10, 201, 199, 202])

# def ssd(p1, p2):
#     return np.sum((p1 - p2) ** 2)

# print(f"ssd(q, p1): {ssd(q, p1)}")
# print(f"ssd(q, p2): {ssd(q, p2)}")

# ratio_p2_p1 = ssd(q, p2) / ssd(q, p1)
# print(f"ratio_p2_p1: {ratio_p2_p1}")

import numpy as np

# A = [[3, 0], [4, 7], [-10, -3]]
# B = [[1, 3], [1, 10], [8, -10]]

A = [[-9, 9], [5, 3], [2, -6]]
B = [[-8, -3], [-10, 5], [0, 0]]

def fitting_a_rigid_transformation(A, B):
    def mu(P: np.ndarray): return P.sum(axis=0) / P.shape[0]

    A, B = np.array(A), np.array(B)
    
    # Center both sets
    A_c, B_c = A - mu(A), B - mu(B)
    
    # Determine rotation
    U, s, Vh = np.linalg.svd(A_c.T @ B_c)
    if np.linalg.det(U @ Vh) < 0:
        U[:, -1] *= -1
    R = np.round(U @ Vh).astype(int)

    # Determine translation
    t = np.round(mu((B.T - R @ A.T).T)).astype(int)

    return R, t


rotation_matrix, translation_vector = fitting_a_rigid_transformation(A, B)
print(rotation_matrix)
print()
print(translation_vector)

