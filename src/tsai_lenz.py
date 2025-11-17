from utils import *

def tsai_lenz(As, Bs):
    def _safe_unit(w, eps=1e-12):
        n = np.linalg.norm(w)
        if n < eps:
            return np.zeros_like(w)
        return w / n

    Arel = []
    Brel = []
    for i in range(len(As) - 1):
        Arel.append(invert_T(As[i]) @ As[i + 1])
        Brel.append(invert_T(Bs[i]) @ Bs[i + 1])

    angle_min = np.deg2rad(2.0)
    pairs = []
    for Ar, Br in zip(Arel, Brel):
        av = log_SO3(Ar[:3, :3])
        bv = log_SO3(Br[:3, :3])
        if np.linalg.norm(av) >= angle_min and np.linalg.norm(bv) >= angle_min:
            pairs.append((Ar, Br))
    if not pairs:
        pairs = list(zip(Arel, Brel))

    n = len(pairs)

    S_rows = []
    v_rows = []
    for Ar, Br in pairs:
        a_vec = log_SO3(Ar[:3, :3])
        b_vec = log_SO3(Br[:3, :3])
        a = _safe_unit(a_vec)
        b = _safe_unit(b_vec)
        S_rows.append(hat(a + b))
        v_rows.append(a - b)

    S = np.vstack(S_rows)
    v = np.concatenate(v_rows)

    x, _, _, _ = np.linalg.lstsq(S, v, rcond=None)

    x_norm = np.linalg.norm(x)
    if x_norm < 1e-12:
        R = np.eye(3)
    else:
        theta = 2.0 * np.arctan(x_norm)
        u = x / x_norm
        R = (np.eye(3) * np.cos(theta)
             + np.sin(theta) * hat(u)
             + (1.0 - np.cos(theta)) * (u.reshape(3, 1) @ u.reshape(1, 3)))
        R = R.T


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
