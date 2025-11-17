import numpy as np
import pandas as pd

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

def compose(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def euler_ZYX_to_R(z, y, x):
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)

    Rz = np.array([[ cz, -sz,  0],
                   [ sz,  cz,  0],
                   [  0,   0,  1]])

    Ry = np.array([[ cy,  0, sy],
                   [  0,  1,  0],
                   [-sy,  0, cy]])

    Rx = np.array([[ 1,  0,   0],
                   [ 0, cx, -sx],
                   [ 0, sx,  cx]])

    return Rz @ Ry @ Rx

def log_SO3(R):
    trc = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    theta = np.arccos(trc)
    if theta < 1e-12:
        return np.zeros(3)
    w = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]]) / (2 * np.sin(theta))
    return theta * w


def hat(w):
    wx, wy, wz = w
    return np.array([[ 0,  -wz,  wy],
                     [ wz,   0, -wx],
                     [-wy,  wx,   0]])


def load_poses_csv(path):
    df = pd.read_csv(
        path,
        sep=r'(?:,\s+|\s+)',
        usecols=range(7),
        header=None,
        engine="python",
        names=["id", "X", "Y", "Z", "RZ", "RY", "RX"]
    )
    return df


def df_to_Ts(df):
    Ts = []
    for _, r in df.iterrows():
        R = euler_ZYX_to_R(
            np.deg2rad(r["RZ"]),
            np.deg2rad(r["RY"]),
            np.deg2rad(r["RX"])
        )
        t = np.array([r["X"], r["Y"], r["Z"]])
        Ts.append(compose(R, t))
    return np.array(Ts)


def summarize_errors(As, Bs, X, Y):
    t_errs = []
    r_errs = []

    for A, B in zip(As, Bs):
        Delta = invert_T(A @ X) @ (Y @ B)

        t_errs.append(np.linalg.norm(Delta[:3, 3]))
        angle = np.linalg.norm(log_SO3(Delta[:3, :3])) * 180.0 / np.pi
        r_errs.append(angle)

    t_errs = np.array(t_errs)
    r_errs = np.array(r_errs)

    def stats(x):
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "rmse": float(np.sqrt(np.mean(x**2))),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x))
        }

    return stats(t_errs), stats(r_errs)


def print_T(name, T):
    print(f"{name} =")
    for i in range(4):
        row = " ".join(f"{T[i, j]:.4f}" for j in range(4))
        print(f"[ {row} ]")


def calculate_Z(As, Bs, X):
    RX = X[:3, :3]
    tX = X[:3, 3]

    M = np.zeros((3, 3))
    for A, B in zip(As, Bs):
        Ra = A[:3, :3]
        Rb = B[:3, :3]
        M += Ra @ RX @ Rb.T
    U, _, Vt = np.linalg.svd(M)
    RZ = U @ Vt
    if np.linalg.det(RZ) < 0:
        U[:, -1] *= -1
        RZ = U @ Vt

    C = np.zeros((3 * len(As), 3))
    d = np.zeros(3 * len(As))
    for i, (A, B) in enumerate(zip(As, Bs)):
        Ra = A[:3, :3]
        ta = A[:3, 3]
        tb = B[:3, 3]
        C[3*i:3*i+3, :] = np.eye(3)
        d[3*i:3*i+3] = ta + Ra @ tX - RZ @ tb
    tZ, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    Z = compose(RZ, tZ)
    return Z
