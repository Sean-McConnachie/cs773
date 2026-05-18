import numpy as np
from q1 import compute_homo

def forward_transform(X, H, yx = True):
    if yx:
        result = np.array([H @ np.array([pt[1], pt[0], 1.0]) for pt in X])
    else:
        result = np.array([H @ np.array([pt[0], pt[1], 1.0]) for pt in X])
    result[:, 0] = result[:, 0] / result[:, 2]
    result[:, 1] = result[:, 1] / result[:, 2]
    return result[:, :2]

def quick_test():
    X = np.array([
        [2, 2],
        [4, 2],
        [4, 6],
        [2, 6]
    ])
    
    Xp = np.array([
        [-2, -2],
        [-4, -1],
        [-1, -6],
        [-5, -6]
    ])

    pairs = [ [X[i], Xp[i]] for i in range(len(X)) ]
    X = np.array([p[0] for p in pairs])
    Xp = np.array([p[1] for p in pairs])
    
    print("X\n", X)
    print("Xp\n", Xp)

    H = compute_homo(pairs)
    Xm = forward_transform(X, H)
    print(f"Xm={Xm}")
    print(f"Xp={Xp}")


quick_test()