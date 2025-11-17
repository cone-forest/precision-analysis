from utils import *

def park_martin(As, Bs):
    Arel = []
    Brel = []
    for i in range(len(As) - 1):
        Arel.append(invert_T(As[i]) @ As[i + 1])
        Brel.append(invert_T(Bs[i]) @ Bs[i + 1])

    pairs = []
    thr = np.deg2rad(2.0)
    for Ar, Br in zip(Arel, Brel):
        if np.linalg.norm(log_SO3(Ar[:3, :3])) >= thr and np.linalg.norm(log_SO3(Br[:3, :3])) >= thr:
            pairs.append((Ar, Br))
    if not pairs:
        pairs = list(zip(Arel, Brel))

    M = np.zeros((3, 3))
    for Ar, Br in pairs:
        a = log_SO3(Ar[:3, :3])
        b = log_SO3(Br[:3, :3])
        M = M + np.outer(b, a)

    E = M.T @ M
    w, U = np.linalg.eigh(E)
    w[w < 1e-15] = 1e-15
    inv_sqrt = U @ np.diag(w ** -0.5) @ U.T
    R = inv_sqrt @ M.T

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    n = len(pairs)
    C = np.zeros((3 * n, 3))
    d = np.zeros(3 * n)
    I = np.eye(3)

    for i, (Ar, Br) in enumerate(pairs):
        Ra = Ar[:3, :3]
        ta = Ar[:3, 3]
        tb = Br[:3, 3]

        C[3*i:3*i+3, :] = I - Ra
        d[3*i:3*i+3] = ta - R @ tb

    t, _, _, _ = np.linalg.lstsq(C, d, rcond=None)

    X = compose(R, t)
    Z = calculate_Z(As, Bs, X)

    return X, Z
