import numpy as np

def compute_homo(matched_corner_pairs):
    X = np.array([p[0] for p in matched_corner_pairs])
    X_ = np.array([p[1] for p in matched_corner_pairs])

    # X = np.array([
    #     [2, 2, 1]
    # ])

    n = len(X)
    A = np.zeros((2*n, 9))

    for i in range(n):
        y, x = X[i][:2]
        y_, x_ = X_[i][:2]
        A[i*2]   = np.array([0, 0, 0, x, y, 1, -y_*x, -y_*y, -y_])
        A[i*2 + 1] = np.array([x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_])
    
    U, D, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H
    
    result = np.array([H @ np.array([pt[0], pt[1], 1]) for pt in X])
    result[:, 0] = result[:, 0] / result[:, 2]
    result[:, 1] = result[:, 1] / result[:, 2]
    result[:, 2] = result[:, 2] / result[:, 2]
    result = result[:, :2]
    return result
    
filtered_matched_corner_pairs = np.load("data/filtered_matched_corner_pairs.npy")
transformation_matrix = np.load("data/transformation_matrix.npy")

H = compute_homo(filtered_matched_corner_pairs)

print(transformation_matrix)
print(H)