from utils import *

def li_wang_wu(As, Bs):
    n = len(As)

    A = np.zeros((12 * n, 24))
    b = np.zeros((12 * n, 1))
    I3 = np.eye(3)

    for i, (A1, B1) in enumerate(zip(As, Bs)):
        Ra = A1[:3, :3]
        Rb = B1[:3, :3]
        ta = A1[:3, 3]
        tb = B1[:3, 3]

        A[12*i:12*i+9, 0:18] = np.hstack([
            np.kron(Ra, I3),
            np.kron(-I3, Rb.T)
        ])

        A[12*i+9:12*i+12, 9:24] = np.hstack([
            np.kron(I3, tb.reshape(1, 3)),
            -Ra,
            I3
        ])

        b[12*i+9:12*i+12, 0] = ta

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x = x.ravel()

    Xr = np.reshape(x[0:9], (3, 3), order='F').T
    U, S, Vt = np.linalg.svd(Xr)
    Xr = U @ Vt
    if np.linalg.det(Xr) < 0:
        U[:, -1] *= -1
        Xr = U @ Vt
    X = compose(Xr, x[18:21])

    Yr = np.reshape(x[9:18], (3, 3), order='F').T
    U, S, Vt = np.linalg.svd(Yr)
    Yr = U @ Vt
    if np.linalg.det(Yr) < 0:
        U[:, -1] *= -1
        Yr = U @ Vt
    Y = compose(Yr, x[21:24])

    return X, Y
