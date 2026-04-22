import numpy as np

# I = np.array([ [24,19,17,20,24],
#                [19,12,11,12,20],
#                [18,10,3,10,19],
#                [20,13,11,13,21],
#                [24,21,18,20,26]])

I = np.array([[35, 29, 25, 19, 18],
              [36, 30, 25, 20, 17],
              [35, 29, 27, 19, 16],
              [36, 30, 26, 21, 16],
              [34, 30, 27, 21, 17]])


def apply_kernel(I: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    knx, kny = kernel.shape[1], kernel.shape[0]
    ret = np.zeros_like(I)
    for y in range(1, I.shape[0] - 1):
        for x in range(1, I.shape[0] - 1):
            patch = I[y-kny//2:y+kny//2+1, x-knx//2:x+knx//2+1]
            v = (patch * kernel).sum()
            ret[y, x] = v
    return ret


Ix = apply_kernel(I, np.array([[-1,0,1]]))
Iy = apply_kernel(I, np.array([[-1,0,1]]).T)
print(f"Ix =\n{Ix}")
print(f"Iy =\n{Iy}")

Ixx = Ix**2
Iyy = Iy**2
Ixy = Ix*Iy

print(f"Ixx =\n{Ixx}")
print(f"Iyy =\n{Iyy}")
print(f"Ixy =\n{Ixy}")

print(f"sum(Ixx) = {Ixx.sum()}")
print(f"sum(Iyy) = {Iyy.sum()}")
print(f"sum(Ixy) = {Ixy.sum()}")

det = Ixx.sum() * Iyy.sum() - Ixy.sum()**2
trace = Ixx.sum() + Iyy.sum()
C = det - 0.04 * trace**2
print(f"det = {det}")
print(f"trace = {trace}")
print(f"C = {C}")


t = np.array([[3,5,11],
              [10,12,8],
              [12,8,1]])
w = np.array([[3,10,12],
              [12,9,4],
              [8,3,1]])
sad = np.abs(t-w).sum()
ssd = ((t-w)**2).sum()
print(f"SAD = {sad}")
print(f"SSD = {ssd}")