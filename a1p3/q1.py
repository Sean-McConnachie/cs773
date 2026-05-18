import numpy as np
from decimal import Decimal, ROUND_HALF_UP

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

def round_matrix(matrix, decimals=3):
    """
    Round a NumPy matrix to a given number of decimal places using
    standard round-half-up rounding (not banker's rounding).
    
    Parameters
    ----------
    matrix : np.ndarray
        Input array of any shape.
    decimals : int
        Number of decimal places to round to (default: 3).
    
    Returns
    -------
    np.ndarray
        Rounded array with the same shape as the input.
    """
    quantizer = Decimal(10) ** -decimals  # e.g. Decimal('0.001') for 3 dp
    
    def _round(x):
        return float(Decimal(str(x)).quantize(quantizer, rounding=ROUND_HALF_UP))
    
    vectorized = np.vectorize(_round, otypes=[np.float64])
    return vectorized(matrix)

if __name__ == "__main__": 
    filtered_matched_corner_pairs = np.load("data/filtered_matched_corner_pairs.npy")
    print(filtered_matched_corner_pairs[0])
    transformation_matrix = np.load("data/transformation_matrix.npy")
    

    # S1 = [(0, 0, 1), (2, 1, 1), (3, 4, 1), (1, 5, 1)]
    # S2 = [(1, 2, 1), (6, 3, 1), (7, 8, 1), (2, 9, 1)]
    # filtered_matched_corner_pairs = [[S1[i][:2][::-1], S2[i][:2][::-1]] for i in range(len(S1)) ]
    print(filtered_matched_corner_pairs)
    H = compute_homo(filtered_matched_corner_pairs)
    
    print(transformation_matrix)
    print(round_matrix(H, decimals=3))
    