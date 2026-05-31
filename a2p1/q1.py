import numpy as np


np.set_printoptions(suppress=True, precision=2, floatmode='fixed')


class ParamsLeft:
    def __init__(self):
        self.dx = 0.001
        self.dy = 0.001
        self.R = np.array([
            [ 0.7, -0.7,  0.05],
            [-0.5, -0.06, 1.0 ],
            [-0.7, -0.7, -0.1 ],
        ])
        self.tx = 5.0
        self.ty = 6.0
        self.tz = 56.0
        self.f = 2.0
        self.sx = 1.0
        self.cx = 1500.0
        self.cy = 1125.0


class ParamsRight:
    def __init__(self):
        self.dx = 0.001
        self.dy = 0.001
        self.R = np.array([
            [ 0.7, -0.7,  0.04],
            [-0.3, -0.09, 1.0 ],
            [-0.7, -0.7, -0.2 ],
        ])
        self.tx = 2.0
        self.ty = 8.0
        self.tz = 57.0
        self.f = 3.0
        self.sx = 1.0
        self.cx = 1500.0
        self.cy = 1125.0


def round_2dp(m):
    return np.where(m > 0, np.floor(m * 100 + 0.5) / 100, np.ceil(m * 100 - 0.5) / 100)


def print_and_round(matricies, message=None):
    returns = []
    if message:
        print(message)
    for m in matricies:
        rounded_m = round_2dp(m)
        print(rounded_m)
        returns.append(rounded_m)
    return returns


def construct_k(tsai):
    fx = tsai.f * tsai.sx / tsai.dx
    fy = -tsai.f / tsai.dy
    return np.array([
        [fx,    0,  tsai.cx,    0],
        [0,     fy, tsai.cy,    0],
        [0,     0,  1,          0]
    ])

def construct_m(R, t):
    m = np.column_stack((R, t))
    m = np.row_stack((m, np.array([[0, 0, 0, 1]])))
    return m


def compute_p_old(k, m):
    return k @ m


def compute_k_new(p_1_old, p_2_old):
    return (p_1_old + p_2_old) / 2


def compute_c1_c2_b(params_1, params_2):
    c1 = -params_1.R.T.dot(params_1.t)
    c2 = -params_2.R.T.dot(params_2.t)
    b = c2 - c1
    return c1, c2, b


def mag(v):
    return np.sqrt(np.sum(v**2))


def compute_r_new(b, r_1):
    v_x = b / mag(b)
    k = r_1[2].T
    v_y = np.cross(k, v_x)
    v_y /= mag(v_y)
    v_z = np.cross(v_x, v_y)
    v_z /= mag(v_z)
    r_new = np.row_stack([v_x, v_y, v_z])
    return r_new


def compute_p_new(k_new, m_new):
    return k_new @ m_new


def tl_3x3(m):
    return m[:3, :3]


def compute_H(p_new, p_old):
    return tl_3x3(p_new) @ np.linalg.inv(tl_3x3(p_old))


def image_rectification(params_1, params_2):
    params_1.t = np.array([params_1.tx, params_1.ty, params_1.tz]).T
    params_2.t = np.array([params_2.tx, params_2.ty, params_2.tz]).T

    k_1 = construct_k(params_1)
    k_2 = construct_k(params_2)
    k_1, k_2 = print_and_round([k_1, k_2], "k_1 and k_2")
    print("="*20)

    m_1 = construct_m(params_1.R, params_1.t)
    m_2 = construct_m(params_2.R, params_2.t)
    m_1, m_2 = print_and_round([m_1, m_2], "m_1 and m_2")
    print("="*20)

    p_1_old = compute_p_old(k_1, m_1)
    p_2_old = compute_p_old(k_2, m_2)
    p_1_old, p_2_old = print_and_round([p_1_old, p_2_old], "p_1 and p_2 before rectification")
    print("="*20)

    k_new = compute_k_new(k_1, k_2)
    k_new = print_and_round([k_new], "k_new")[0]
    print("="*20)

    c1, c2, b = compute_c1_c2_b(params_1, params_2)
    c1, c2, b = print_and_round([c1, c2, b], "c1, c2 and b")
    print("="*20)

    r_new = compute_r_new(b, params_1.R)
    r_new = print_and_round([r_new], "R_new")[0]
    print("="*20)

    m_1_new = construct_m(r_new, params_1.t)
    m_2_new = construct_m(r_new, params_2.t)
    m_1_new, m_2_new = print_and_round([m_1_new, m_2_new], "m_1_new and m_2_new")
    print("="*20)

    p_1_new = compute_p_new(k_new, m_1_new)
    p_2_new = compute_p_new(k_new, m_2_new)
    p_1_new, p_2_new = print_and_round([p_1_new, p_2_new], "p_1 and p_2 after rectification")
    print("="*20)

    H_1 = compute_H(p_1_new, p_1_old)
    H_2 = compute_H(p_2_new, p_2_old)
    H_1, H_2 = print_and_round([H_1, H_2], "H_1 and H_2")
    print("="*20)


if __name__ == "__main__":
    params_left = ParamsLeft()
    params_right = ParamsRight()
    image_rectification(params_left, params_right)
