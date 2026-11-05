"""Conventional linear-inverse reconstructions for ghost imaging: pseudoinverse and TV.

Measurement model (noise-free, as in the Phase 2 simulation):  B = P O,
P: N x D binary patterns (D = 64*64), O: object vector, B: bucket values.

Pseudoinverse:  O_pinv = P^+ B  via the thin SVD P = U S V^T
    N <  D: minimum-norm solution  V S^-1 U^T B  (lies in the row space of P)
    N >= D: ordinary least squares (unique if P has full column rank)

TV (compressed sensing), solved by FISTA with an FGP proximal step (Beck & Teboulle 2009):
    min_O  0.5 || W (P O - B) ||^2 + lam * TV_iso(O),   O in [0, 1]
with W = (P P^T)^(-1/2) for N < D (whitened residuals) and W = (P^T P)^(-1/2)-equivalent for
N >= D, which makes the data term 0.5 || V^T (O - O_pinv) ||^2 (N < D) or 0.5 || O - O_ls ||^2
(N >= D). For noise-free data the set of data-consistent images is identical to that of the
unweighted 0.5||P O - B||^2; only the scale of lam differs. Gradient Lipschitz constant = 1.
Only P and B are used; the ground truth never enters.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- linear algebra
def svd_patterns(p: np.ndarray):
    """Thin SVD of the N x D pattern matrix (float64). Returns U, s, Vt."""
    return np.linalg.svd(p.astype(np.float64), full_matrices=False)


def pinv_solve(u: np.ndarray, s: np.ndarray, vt: np.ndarray, b: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """O = V S^-1 U^T B for B of shape (N, M) -> (M, D). Singular values below rcond*s_max are dropped."""
    keep = s > rcond * s[0]
    coef = (u[:, keep].T @ b) / s[keep][:, None]
    return (vt[keep].T @ coef).T


# ---------------------------------------------------------------- TV operators (batched, (M, H, W))
def grad(x: np.ndarray):
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
    gy[:, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
    return gx, gy


def div(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Negative adjoint of grad: <grad u, p> = -<u, div p>."""
    dx = np.zeros_like(px)
    dx[:, :, 0] = px[:, :, 0]
    dx[:, :, 1:-1] = px[:, :, 1:-1] - px[:, :, :-2]
    dx[:, :, -1] = -px[:, :, -2]
    dy = np.zeros_like(py)
    dy[:, 0, :] = py[:, 0, :]
    dy[:, 1:-1, :] = py[:, 1:-1, :] - py[:, :-2, :]
    dy[:, -1, :] = -py[:, -2, :]
    return dx + dy


def tv_iso(x: np.ndarray) -> np.ndarray:
    gx, gy = grad(x)
    return np.sqrt(gx ** 2 + gy ** 2).sum(axis=(1, 2))


def prox_tv_box(v: np.ndarray, w: float, n_iter: int = 30, state=None):
    """FGP: argmin_u 0.5||u - v||^2 + w TV_iso(u), u in [0, 1]. Returns (u, dual state)."""
    if w <= 0:
        return np.clip(v, 0, 1), state
    if state is None:
        px, py = np.zeros_like(v), np.zeros_like(v)
    else:
        px, py = state
    rx, ry = px.copy(), py.copy()
    t = 1.0
    step = 1.0 / (8.0 * w)
    for _ in range(n_iter):
        u = np.clip(v + w * div(rx, ry), 0, 1)
        gx, gy = grad(u)
        qx, qy = rx + step * gx, ry + step * gy
        nrm = np.maximum(1.0, np.sqrt(qx ** 2 + qy ** 2))
        qx, qy = qx / nrm, qy / nrm
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        rx, ry = qx + ((t - 1) / tn) * (qx - px), qy + ((t - 1) / tn) * (qy - py)
        px, py, t = qx, qy, tn
    return np.clip(v + w * div(px, py), 0, 1), (px, py)


def tv_reconstruct(x_pinv: np.ndarray, vt: np.ndarray | None, lam: float, shape=(64, 64),
                   n_outer: int = 300, n_inner: int = 30, track: bool = False):
    """FISTA for 0.5||V^T(x - x_pinv)||^2 + lam TV(x), x in [0,1]  (vt=None: data term 0.5||x - x_pinv||^2).

    x_pinv: (M, D) pseudoinverse / least-squares solutions. Returns (M, H, W) and an optional trace.
    """
    m = x_pinv.shape[0]
    x0 = x_pinv.reshape(m, *shape)
    x = np.clip(x0, 0, 1)
    y, t, state, trace = x.copy(), 1.0, None, []

    def data_grad(z):
        d = (z - x0).reshape(m, -1)
        if vt is not None:
            d = (d @ vt.T) @ vt
        return d.reshape(z.shape)

    for k in range(n_outer):
        x_new, state = prox_tv_box(y - data_grad(y), lam, n_inner, state)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = x_new + ((t - 1) / tn) * (x_new - x)
        x, t = x_new, tn
        if track and (k % 10 == 0 or k == n_outer - 1):
            d = (x - x0).reshape(m, -1)
            dd = d @ vt.T if vt is not None else d
            trace.append({"iter": k + 1, "objective_mean": float((0.5 * (dd ** 2).sum(1) + lam * tv_iso(x)).mean())})
    return x, trace


# ---------------------------------------------------------------- noise-aware solvers (Phase 4A)
def tikhonov_from_svd(u: np.ndarray, s: np.ndarray, vt: np.ndarray, utb: np.ndarray, lam: float) -> np.ndarray:
    """argmin ||P O - B||^2 + lam ||O||^2 = V diag(s / (s^2 + lam)) U^T B.  utb = U^T B (k, M) -> (M, D).

    lam = 0 gives the pseudoinverse (minimum-norm / least-squares) solution.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(s > 1e-12 * s[0], s / (s ** 2 + lam), 0.0)
    return (vt.T @ (f[:, None] * utb)).T


def gradient_matrix(h: int = 64, w: int = 64):
    """Sparse forward-difference operator G (2HW x HW), Neumann boundary; matches grad()."""
    import scipy.sparse as sp
    def d1(n):
        m = sp.diags([-np.ones(n), np.ones(n - 1)], [0, 1], shape=(n, n), format="lil")
        m[n - 1, n - 1] = 0.0
        return m.tocsr()
    gx = sp.kron(sp.identity(h), d1(w))          # along columns (x)
    gy = sp.kron(d1(h), sp.identity(w))          # along rows (y)
    return sp.vstack([gx, gy]).tocsr()


class TVADMM:
    """min_O 0.5 ||A O - b||^2 + lam * TV_iso(O),  O in [0, 1],  A = 2 P / sqrt(N), b = 2 B / sqrt(N).

    This is the unweighted least-squares data term (the Gaussian maximum-likelihood term for
    i.i.d. bucket noise), rescaled by 4/N so that lam is comparable across N. ADMM splits
    z = G O (isotropic shrinkage) and w = O (box); the O-update solves
    (A^T A + rho (G^T G + I)) O = rhs with a Cholesky factor cached per (P_N, rho).
    """

    def __init__(self, p: np.ndarray, rho: float, shape=(64, 64)):
        from scipy.linalg import cho_factor
        n = p.shape[0]
        a = p.astype(np.float64) * (2.0 / np.sqrt(n))
        self.a, self.rho, self.shape = a, rho, shape
        self.g = gradient_matrix(*shape)
        gtg = (self.g.T @ self.g).toarray()
        self.chol = cho_factor(a.T @ a + rho * (gtg + np.eye(a.shape[1])), lower=True, check_finite=False)

    def solve(self, b: np.ndarray, lam: float, n_iter: int = 300, x0: np.ndarray | None = None, track: bool = False):
        """b: raw buckets (N, M). Returns (M, H, W) and an optional objective trace."""
        from scipy.linalg import cho_solve
        n, d = self.a.shape
        bb = b * (2.0 / np.sqrt(n))
        atb = self.a.T @ bb                                          # (D, M)
        m = bb.shape[1]
        x = np.clip(x0, 0, 1) if x0 is not None else np.full((d, m), 0.5)
        gx = self.g @ x
        z, u = gx.copy(), np.zeros_like(gx)
        w, v = x.copy(), np.zeros_like(x)
        trace = []
        half = d
        for k in range(n_iter):
            rhs = atb + self.rho * (self.g.T @ (z - u)) + self.rho * (w - v)
            x = cho_solve(self.chol, rhs, check_finite=False)
            gx = self.g @ x
            q = gx + u
            mag = np.sqrt(q[:half] ** 2 + q[half:] ** 2)
            shrink = np.maximum(0.0, 1.0 - (lam / self.rho) / np.maximum(mag, 1e-12)) if lam > 0 else np.ones_like(mag)
            z = np.concatenate([q[:half] * shrink, q[half:] * shrink])
            w = np.clip(x + v, 0, 1)
            u += gx - z
            v += x - w
            if track and (k % 10 == 0 or k == n_iter - 1):
                xr = np.clip(x, 0, 1)
                gr = self.g @ xr
                tv = np.sqrt(gr[:half] ** 2 + gr[half:] ** 2).sum(0)
                obj = 0.5 * ((self.a @ xr - bb) ** 2).sum(0) + lam * tv
                trace.append({"iter": k + 1, "objective_mean": float(obj.mean()),
                              "primal_residual": float(np.linalg.norm(gx - z) / np.sqrt(m))})
        return w.T.reshape(m, *self.shape), trace
