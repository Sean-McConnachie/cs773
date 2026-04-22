import numpy as np

I = """
11
8
2
6
12
18
13
8
2
7
I =	23
16
13
6
1
26
22
16
11
7
31
28
22
16
12
"""

def parse_matrix(m, size: int=5):
    vals = [int(v.split("\t")[-1]) for v in m.strip().split("\n")]
    print(vals)
    M = []
    for i in range(size):
        M.append(vals[i*size:(i+1)*size])
    print(M)
    return np.array(M)


def apply_kernel(I, kernel):
    k_size = kernel.shape[0]
    offset = k_size // 2
    result = np.zeros_like(I)
    for i in range(offset, I.shape[0] - offset):
        for j in range(offset, I.shape[1] - offset):
            region = I[i-offset:i+offset+1, j-offset:j+offset+1]
            result[i, j] = np.sum(region * kernel)
    return result


I = parse_matrix(I, size=5)
Ix = apply_kernel(I, np.array([ [0, 0, 0],
                                [-1, 0, 1],
                                [0, 0, 0]]))
Iy = apply_kernel(I, np.array([ [0,  1, 0],
                                [0,  0, 0],
                                [0, -1, 0]]))

print(f"Ix = {Ix}")
print(f"Iy = {Iy}")

Ixx = ((Ix * Ix) / 9).sum()
Iyy = ((Iy * Iy) / 9).sum()
Ixy = ((Ix * Iy) / 9).sum()  # Attempt 4

print(f"Ixx = {round(Ixx, 2)}")
print(f"Iyy = {round(Iyy, 2)}")
print(f"Ixy = {round(Ixy, 2)}")

C = (Ixx * Iyy - Ixy * Ixy) - 0.04 * (Ixx + Iyy)**2
print(f"C = {round(C, 2)}")