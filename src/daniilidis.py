from utils import *


def hom2quar(H):
    R = H[:3, :3]
    t = H[:3, 3]

    a = log_SO3(R)
    theta = np.linalg.norm(a)
    if theta < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        l = a / theta
        q = np.array([
            np.cos(theta / 2.0),
            np.sin(theta / 2.0) * l[0],
            np.sin(theta / 2.0) * l[1],
            np.sin(theta / 2.0) * l[2]
        ])

    qprime = 0.5 * qmult(np.array([0.0, t[0], t[1], t[2]]), q)

    dq = np.zeros((4, 2))
    dq[:, 0] = q
    dq[:, 1] = qprime
    return dq


def quar2hom(dq):
    q = dq[:, 0]
    qe = dq[:, 1]

    nrm = np.linalg.norm(q)
    if nrm < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q = q / nrm

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    R = np.array([
        [1 - 2*q2*q2 - 2*q3*q3, 2*(q1*q2 - q3*q0),     2*(q1*q3 + q2*q0)],
        [2*(q1*q2 + q3*q0),     1 - 2*q1*q1 - 2*q3*q3, 2*(q2*q3 - q1*q0)],
        [2*(q1*q3 - q2*q0),     2*(q2*q3 + q1*q0),     1 - 2*q1*q1 - 2*q2*q2]
    ])

    qc = q.copy()
    qc[1:] = -qc[1:]

    tq = 2.0 * qmult(qe, qc)
    t = tq[1:]

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def qmult(p, q):
    w1 = p[0]
    x1 = p[1]
    y1 = p[2]
    z1 = p[3]

    w2 = q[0]
    x2 = q[1]
    y2 = q[2]
    z2 = q[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def daniilidis(As, Bs):
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

    n = len(pairs)
    T = np.zeros((6 * n, 8))

    for i, (A1, B1) in enumerate(pairs):
        a = hom2quar(A1)
        b = hom2quar(B1)

        a1 = a[:, 0]; a2 = a[:, 1]
        b1 = b[:, 0]; b2 = b[:, 1]

        v_a1 = a1[1:]; v_b1 = b1[1:]
        v_a2 = a2[1:]; v_b2 = b2[1:]

        upper = np.hstack([
            (v_a1 - v_b1).reshape(3, 1),
            hat(v_a1 + v_b1),
            np.zeros((3, 4))
        ])
        lower = np.hstack([
            (v_a2 - v_b2).reshape(3, 1),
            hat(v_a2 + v_b2),
            (v_a1 - v_b1).reshape(3, 1),
            hat(v_a1 + v_b1)
        ])

        T[6*i:6*i+3, :] = upper
        T[6*i+3:6*i+6, :] = lower

    U, s, Vt = np.linalg.svd(T)
    V = Vt.T

    u1 = V[0:4, 6]; v1 = V[4:8, 6]
    u2 = V[0:4, 7]; v2 = V[4:8, 7]

    a = float(u1.T @ v1)
    b = float(u1.T @ v2 + u2.T @ v1)
    c = float(u2.T @ v2)

    disc = b*b - 4*a*c
    if disc < 0:
        disc = 0.0
    sqrt_disc = np.sqrt(disc)

    if abs(a) < 1e-15:
        s_cands = [(-c / b) if abs(b) > 1e-15 else 0.0]
    else:
        s_cands = [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]

    u1u1 = float(u1.T @ u1)
    u1u2 = float(u1.T @ u2)
    u2u2 = float(u2.T @ u2)

    best_val = -np.inf
    s_opt = s_cands[0]
    for sc in s_cands:
        val = sc*sc*u1u1 + 2*sc*u1u2 + u2u2
        if val > best_val:
            best_val = val
            s_opt = sc

    L2 = 1.0 if best_val <= 1e-18 else np.sqrt(1.0 / best_val)
    L1 = s_opt * L2

    q8 = L1 * V[:, 6] + L2 * V[:, 7]

    dq = np.zeros((4, 2))
    dq[:, 0] = q8[0:4]
    dq[:, 1] = q8[4:8]

    X = quar2hom(dq)
    Z = calculate_Z(As, Bs, X)

    return X, Z

