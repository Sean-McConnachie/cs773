import numpy as np


class H3LeftTsaiCali():
    def __init__(self):
        self.dx = 0.00155
        self.dy = 0.00155
        self.R = np.array([[  0.69358669,  0.05336124,  -0.7183941],
                           [  0.13580093, -0.98883485,  0.06134937],
                           [-0.707101922, -0.14011019, -0.69309163]])
        self.tx = 2.0199458442709664
        self.ty = 9.952670170898207
        self.tz = 58.6894307233427
        self.f = 2.708689448955338
        self.sx = 1.0018491172000161
        self.cx = 1500
        self.cy = 1125

class H3RightTsaiCali():
    def __init__(self):
        self.dx = 0.00155
        self.dy = 0.00155
        self.R = np.array([[ 0.69863844,  0.03899342, -0.71441153],
                           [ 0.14357909, -0.98561574,  0.08914402],
                           [-0.70066037, -0.16485427, -0.69418882]])
        self.tx = -1.6065117005016987
        self.ty = 8.054880536832522
        self.tz = 60.16429265514225
        self.f = 2.7729407250581732
        self.sx = 1.0001142711887294
        self.cx = 1500
        self.cy = 1125


def forward_projection(corner_list, tsai):
    K = np.eye(3)
    K[0] = [tsai.sx / tsai.dx, 0, tsai.cx]
    K[1] = [0, 1 / tsai.dy, tsai.cy]

    I = np.zeros((3, 4))
    I[0, 0] = tsai.f
    I[1, 1] = tsai.f
    I[2, 2] = 1

    Rt = np.eye(4)
    Rt[:3, :3] = tsai.R
    Rt[:3, 3] = [tsai.tx, tsai.ty, tsai.tz]

    ones = np.ones((corner_list.shape[0], 1))
    corner_list_homo = np.hstack((corner_list, ones))

    proj = (K @ I @ Rt @ corner_list_homo.T).T
    proj = proj / proj[:, 2:3]
    return proj[:, :2]


if __name__ == "__main__":
    left_cali = H3LeftTsaiCali()
    right_cali = H3RightTsaiCali()

    corner_list_left = np.loadtxt("data/H3_Cube_Left_v2.csv", skiprows=1, usecols=(0, 1, 2), delimiter=",")
    corner_list_right = np.loadtxt("data/H3_Cube_Right_v2.csv", skiprows=1, usecols=(0, 1, 2), delimiter=",")

    left_proj = forward_projection(corner_list_left, left_cali)
    right_proj = forward_projection(corner_list_right, right_cali)

    expected_left = np.load("data/H3_left_projected_2D_points.npy")
    expected_right = np.load("data/H3_right_projected_2D_points.npy")

    print(expected_left - left_proj)