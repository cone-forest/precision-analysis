from utils import *

def shah(As, Bs):
    n = len(As)

    T9 = np.zeros((9, 9))
    for i in range(n):
        Ra = As[i][:3, :3]
        Rb = Bs[i][:3, :3]
        T9 = T9 + np.kron(Rb, Ra)

    U, S, Vt = np.linalg.svd(T9)
    x = Vt.T[:, 0]
    y = U[:, 0]

    Xr = np.reshape(x, (3, 3), order='F')
    detX = np.linalg.det(Xr)
    Xr = (np.sign(detX) / (np.abs(detX) ** (1/3))) * Xr
    Ux, Sx, Vtx = np.linalg.svd(Xr)
    Xr = Ux @ Vtx

    Yr = np.reshape(y, (3, 3), order='F')
    detY = np.linalg.det(Yr)
    Yr = (np.sign(detY) / (np.abs(detY) ** (1/3))) * Yr
    Uy, Sy, Vty = np.linalg.svd(Yr)
    Yr = Uy @ Vty

    A_lin = np.zeros((3 * n, 6))
    b_lin = np.zeros((3 * n, 1))

    vecY = np.reshape(Yr, (9, 1), order='F')

    for i in range(n):
        Ra = As[i][:3, :3]
        ta = As[i][:3, 3].reshape(3, 1)
        tb = Bs[i][:3, 3].reshape(3, 1)

        A_lin[3*i:3*i+3, 0:3] = -Ra
        A_lin[3*i:3*i+3, 3:6] = np.eye(3)

        K = np.kron(tb.T, np.eye(3))
        b_lin[3*i:3*i+3, 0:1] = ta - K @ vecY

    t_sol, _, _, _ = np.linalg.lstsq(A_lin, b_lin, rcond=None)
    tX = t_sol[0:3].reshape(3)
    tY = t_sol[3:6].reshape(3)

    X = compose(Xr, tX)
    Y = compose(Yr, tY)
    return X, Y