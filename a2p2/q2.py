import numpy as np
import csv

np.set_printoptions(suppress=True, precision=6)


def get_dx_dy(camera_type):
    if camera_type == "H3":
        sx = 0.00155
        sy = 0.00155
    elif camera_type == "W3":
        sx = 0.001691
        sy = 0.001663
    else:
        raise ValueError()
    return sx, sy


def upward_c1_xp_yp(sx, dx, cx, dy, cy, u, v):
    xp = sx * dx * (u - cx)
    yp = dy * (v - cy)
    return xp, yp


def compute_xp_yp(corner_list, callback):
    """
    Return X[0], Y[1], Z[2], u'[3], v'[4], xp[5], yp[6]
    """
    new_list = []
    for row in corner_list:
        row = list(row)
        row.extend(callback(row[3], row[4]))
        new_list.append(row)
    return np.array(new_list)


def construct_and_solve_linear_equations(corner_list):
    A = np.zeros((len(corner_list), 7), dtype=np.float64)
    b = np.zeros(len(corner_list), dtype=np.float64)
    for i, (X, Y, Z, u, v, xp, yp) in enumerate(corner_list):
        A[i] = np.array([yp * X, yp * Y, yp * Z, yp, -xp * X, -xp * Y, -xp * Z])
        b[i] = xp
    L = np.linalg.pinv(A) @ b
    return L


def compute_ty_abs(L):
    a5 = L[4]
    a6 = L[5]
    a7 = L[6]
    ty = 1 / np.sqrt(a5**2 + a6**2 + a7**2)
    return ty


def compute_sx(L, ty_abs):
    a1 = L[0]
    a2 = L[1]
    a3 = L[2]
    return ty_abs * np.sqrt(a1**2 + a2**2 + a3**2)


def compute_ty(corner_list, L, ty_abs, sx):
    a1, a2, a3, a4, a5, a6, a7 = L

    r11, r12, r13 = a1 * ty_abs / sx, a2 * ty_abs / sx, a3 * ty_abs / sx
    r21, r22, r23 = a5 * ty_abs, a6 * ty_abs, a7 * ty_abs
    tx = a4 * ty_abs / sx

    # Reference point whose image lies furthest from the center
    idx = np.argmax(corner_list[:, 5] ** 2 + corner_list[:, 6] ** 2)
    X, Y, Z, u, v, xp, yp = corner_list[idx]

    Sx = r11 * X + r12 * Y + r13 * Z + tx
    Sy = r21 * X + r22 * Y + r23 * Z + ty_abs

    if np.sign(Sx) == np.sign(xp) and np.sign(Sy) == np.sign(yp):
        return ty_abs
    return -ty_abs


def compute_R_and_tx(L, ty, sx):
    a1, a2, a3, a4, a5, a6, a7 = L
    r11, r12, r13 = a1 * ty / sx, a2 * ty / sx, a3 * ty / sx
    r21, r22, r23 = a5 * ty, a6 * ty, a7 * ty
    tx = a4 * ty / sx

    r31l = r12 * r23 - r13 * r22
    r32l = -(r11 * r23 - r13 * r21)
    r33l = r11 * r22 - r12 * r21

    lam = np.sqrt(1 / (r31l**2 + r32l**2 + r33l**2))
    r31, r32, r33 = r31l * lam, r32l * lam, r33l * lam

    R = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])
    return R, tx


def compute_f_tz(corner_list, R, ty):
    r11, r12, r13 = R[0]
    r21, r22, r23 = R[1]
    r31, r32, r33 = R[2]

    M = np.zeros((len(corner_list), 2), dtype=np.float64)
    A = np.zeros(len(corner_list), dtype=np.float64)
    for i, (X, Y, Z, u, v, xp, yp) in enumerate(corner_list):
        M[i] = np.array([r21 * X + r22 * Y + r23 * Z + ty, -yp])
        A[i] = yp * r31 * X + yp * r32 * Y + yp * r33 * Z
    f, tz = np.linalg.pinv(M) @ A
    return f, tz


def Tsai_calibration(corner_list, camera_type, image_width, image_height):
    # /home/dude-desktop/Sync/mai/cs773/notes/topics/tsai-calibration.md
    sx = 1
    cx = image_width / 2
    cy = image_height / 2
    dx, dy = get_dx_dy(camera_type)

    # Step 1
    corner_list = compute_xp_yp(corner_list, lambda u, v: upward_c1_xp_yp(sx, dx, cx, dy, cy, u, v))

    # Step 2
    L = construct_and_solve_linear_equations(corner_list)

    # Step 3
    ty_abs = compute_ty_abs(L)

    # Step 4
    sx = compute_sx(L, ty_abs)

    # Step 5
    ty = compute_ty(corner_list, L, ty_abs, sx)

    # Step 6-7
    R, tx = compute_R_and_tx(L, ty, sx)

    # Step 8
    f, tz = compute_f_tz(corner_list, R, ty)

    # Construct Rt
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = [tx, ty, tz]
    
    # Only assign values to the corresponding variables, do not alter anything else!
    required_parameters = {
        'a1': L[0],
        'a2': L[1],
        'a3': L[2],
        'a4': L[3],
        'a5': L[4],
        'a6': L[5],
        'a7': L[6],
        'sx': sx,
        'tx': tx,
        'ty': ty,
        'tz': tz,
        'f': f,
        'Rt': Rt
    }
    
    return required_parameters


if __name__ == "__main__":
    # H3 left camera calibration parameters:
    H3_left_params = {'a1': np.float64(0.06981736602157737),
     'a2': np.float64(0.005371413530277033),
     'a3': np.float64(-0.07231451246988359),
     'a4': np.float64(0.20333045565922497),
     'a5': np.float64(0.01364467314942732),
     'a6': np.float64(-0.09935372454521607),
     'a7': np.float64(0.006164111603383542),
     'sx': np.float64(1.0018491172000161),
     'tx': np.float64(2.0199458442709664),
     'ty': np.float64(9.952670170898207),
     'tz': np.float64(58.6894307233427),
     'f': np.float64(2.708689448955338),
     'Rt': np.array([[ 0.69358669,  0.05336124, -0.7183941 ,  2.01994584],
           [ 0.13580093, -0.98883485,  0.06134937,  9.95267017],
           [-0.70710192, -0.14011019, -0.69309163, 58.68943072],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])}

    # H3 right camera calibration parameters:
    H3_right_params = {'a1': np.float64(0.08674471026121204),
     'a2': np.float64(0.004841521400815934),
     'a3': np.float64(-0.08870313665582227),
     'a4': np.float64(-0.19946854222809302),
     'a5': np.float64(0.017825104604862394),
     'a6': np.float64(-0.12236255230362712),
     'a7': np.float64(0.011067081760180598),
     'sx': np.float64(1.0001142711887294),
     'tx': np.float64(-1.6065117005016987),
     'ty': np.float64(8.054880536832522),
     'tz': np.float64(60.16429265514225),
     'f': np.float64(2.7729407250581732),
     'Rt': np.array([[ 0.69863844,  0.03899342, -0.71441153, -1.6065117 ],
           [ 0.14357909, -0.98561574,  0.08914402,  8.05488054],
           [-0.70066037, -0.16485427, -0.69418882, 60.16429266],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])}
    
    # Run H3 Camera Calibration
    image_width = 3000
    image_height = 2250

    H3_left_corner_list = np.genfromtxt("data/H3_Cube_Left_v2.csv", delimiter=",", skip_header=1)
    H3_right_corner_list = np.genfromtxt("data/H3_Cube_Right_v2.csv", delimiter=",", skip_header=1)

    left_required_parameters = Tsai_calibration(H3_left_corner_list, 'H3', image_width, image_height)
    right_required_parameters = Tsai_calibration(H3_right_corner_list, 'H3', image_width, image_height)

    print(f"left:\n{left_required_parameters}")
    print(f"right:\n{right_required_parameters}")

    exit(0)
    # Verify Student's Answers
    print(verify_H3_left_calib(left_required_parameters))
    print(verify_H3_right_calib(right_required_parameters))