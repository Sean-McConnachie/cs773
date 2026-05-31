import numpy as np

np.set_printoptions(suppress=True, precision=8, floatmode='fixed')

class H3LeftTsaiCali():
    def __init__(self):
        self.dx = 0.00155
        self.dy = 0.00155
        self.R = np.array([[ 0.69358669,  -0.7183941,  0.05336124],
                           [-0.13580093, -0.06134937,  0.98883485],
                           [-0.70710192, -0.69309163, -0.14011019]])
        self.tx = 2.0199458439516227
        self.ty = -9.952670170284382
        self.tz = 58.68943072204245
        self.f = 2.708689448793061
        self.sx = 1.0018491172850652
        self.cx = 1500
        self.cy = 1125

class H3RightTsaiCali():
    def __init__(self):
        self.dx = 0.00155
        self.dy = 0.00155
        self.R = np.array([[ 0.69863844, -0.71441153,  0.03899342],
                           [-0.14357909, -0.08914402,  0.98561574],
                           [-0.70066037, -0.69418882, -0.16485427]])
        self.tx = -1.6065117014923782
        self.ty = -8.054880535641873
        self.tz = 60.16429266232803
        self.f = 2.772940725368117
        self.sx = 1.0001142712275077
        self.cx = 1500
        self.cy = 1125


def add_t_to_tsai(tsai):
    tsai.t = np.array([tsai.tx, tsai.ty, tsai.tz]).T

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

def mag(u):
    return np.sqrt(np.sum(u**2))
    
def tl_3x3(m):
    return m[:3, :3]

def rectification(tsai_l, tsai_r):
    # https://canvas.auckland.ac.nz/courses/140820/files/17663884?module_item_id=2948043
    add_t_to_tsai(tsai_l)
    add_t_to_tsai(tsai_r)
    
    # requirements
    k_l, k_r = construct_k(tsai_l), construct_k(tsai_r)
    m_l, m_r = construct_m(tsai_l.R, tsai_l.t), construct_m(tsai_r.R, tsai_r.t)
    
    # step 1
    p_l = k_l @ m_l
    p_r = k_r @ m_r
    
    # step 2
    k_new = (k_l + k_r) / 2
    
    # step 3
    c_l = -tsai_l.R.T.dot(tsai_l.t)
    c_r = -tsai_r.R.T.dot(tsai_r.t)
    b = c_r - c_l
    
    # step 4
    V_x = b / mag(b)
    V_y = np.cross(tsai_l.R[2, :], V_x)
    V_y = V_y / mag(V_y)
    V_z = np.cross(V_x, V_y)
    V_z = V_z / mag(V_z)
    R_new = np.row_stack((V_x, V_y, V_z))  # try column stack if it doesn't work
    
    # step 5
    m_l_new, m_r_new = construct_m(R_new, tsai_l.t), construct_m(R_new, tsai_r.t)
    
    # step 6
    p_l_new, p_r_new = k_new @ m_l_new, k_new @ m_r_new
    
    # step 7
    H_l = tl_3x3(p_l_new) @ np.linalg.inv(tl_3x3(p_l))
    H_r = tl_3x3(p_r_new) @ np.linalg.inv(tl_3x3(p_r))
    
    return H_l, H_r


def verify_expected_H(actual_H_l, actual_H_r, expected_H_l, expected_H_r, tol=1e-6, verbose=True):
    """Compare actual vs expected homographies.

    Returns True if both matrices match within absolute tolerance `tol`.
    Prints max absolute differences when `verbose` is True.
    """
    actual_H_l = np.asarray(actual_H_l)
    actual_H_r = np.asarray(actual_H_r)
    expected_H_l = np.asarray(expected_H_l)
    expected_H_r = np.asarray(expected_H_r)

    if actual_H_l.shape != expected_H_l.shape or actual_H_r.shape != expected_H_r.shape:
        if verbose:
            print("Shape mismatch:")
            print(" actual left:", actual_H_l.shape, "expected left:", expected_H_l.shape)
            print(" actual right:", actual_H_r.shape, "expected right:", expected_H_r.shape)
        return False

    ok_l = np.allclose(actual_H_l, expected_H_l, atol=tol, rtol=0)
    ok_r = np.allclose(actual_H_r, expected_H_r, atol=tol, rtol=0)

    if verbose:
        if ok_l and ok_r:
            print(f"Both homographies match expected within tol={tol}")
        else:
            if not ok_l:
                diff_l = actual_H_l - expected_H_l
                print("Left homography differs. max abs diff:", np.max(np.abs(diff_l)))
                print(diff_l)
            if not ok_r:
                diff_r = actual_H_r - expected_H_r
                print("Right homography differs. max abs diff:", np.max(np.abs(diff_r)))
                print(diff_r)

    return bool(ok_l and ok_r)


if __name__ == "__main__":

    expected_H_l = np.array([
        [ 0.6596, 0.0763, -3083.3251],
        [-0.3281, 0.9803, -1670.0809],
        [-0.0002,      0,    -0.6304]
    ])
    expected_H_r = np.array([
        [ 0.6513, 0.093,  -3081.063],
        [-0.3048, 0.977, -1660.0774],
        [-0.0002,     0,    -0.6603]
    ])

    tsai_l = H3LeftTsaiCali()
    tsai_r = H3RightTsaiCali()
    H_l, H_r = rectification(tsai_l, tsai_r)

    # verify and exit with status
    ok = verify_expected_H(H_l, H_r, expected_H_l, expected_H_r, tol=1e-6)
    if not ok:
        raise SystemExit(1)

