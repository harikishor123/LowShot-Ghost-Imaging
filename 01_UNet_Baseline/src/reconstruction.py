"""Correlation-based (conventional) ghost image reconstruction.

    GI_N(x, y) = < (B_i - <B>) * (P_i(x, y) - <P(x, y)>) >_{i=1..N}

where <.> is the plain average over the first N shots (1/N, as in the standard
definition). This is the sample covariance between the bucket signal and each
pixel's illumination.

Normalisation to [0, 1] (two documented options; neither alters GI itself):

  * "calibrated" (primary): for i.i.d. patterns with per-pixel variance s2,
        E[GI_N(x)] = s2 * (N - 1)/N * O(x),
    so GI_N / s2 is an (asymptotically) unbiased estimate of O in the object's
    own units. We divide by the theoretical s2 = p(1-p) and clip to [0, 1].
    No per-image fitting is involved.
  * "minmax" (secondary): (GI - min) / (max - min). Common in the GI literature
    for display, but it depends on the noisiest extreme pixels.
"""

from __future__ import annotations

import numpy as np


def cgi_reconstruct(patterns: np.ndarray, buckets: np.ndarray, n_shots: int) -> np.ndarray:
    """Raw CGI image from the FIRST n_shots patterns/buckets (float64, H x W)."""
    if not 1 < n_shots <= patterns.shape[0]:
        raise ValueError(f"n_shots must be in (1, {patterns.shape[0]}], got {n_shots}")
    h, w = patterns.shape[1:]
    p = patterns[:n_shots].reshape(n_shots, -1).astype(np.float64)
    b = buckets[:n_shots].astype(np.float64)
    db = b - b.mean()                       # B_i - <B>
    dp = p - p.mean(axis=0, keepdims=True)  # P_i(x,y) - <P(x,y)>
    gi = (db @ dp) / n_shots                # < dB * dP >
    return gi.reshape(h, w)


def normalize_calibrated(gi: np.ndarray, pattern_var: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (unclipped estimate GI/s2, clipped-to-[0,1] estimate)."""
    est = gi / pattern_var
    return est, np.clip(est, 0.0, 1.0)


def dgi_reconstruct(patterns: np.ndarray, buckets: np.ndarray, refs: np.ndarray,
                    n_shots: int) -> tuple[np.ndarray, float]:
    """Differential ghost imaging (Ferri et al., PRL 104, 253603, 2010), first n_shots.

        S_i       = B_i - (<B>/<R>) R_i
        DGI_N(x)  = < S_i (P_i(x) - <P(x)>) >  ==  <B P(x)> - (<B>/<R>) <R P(x)>

    (the two forms are identical because <S> = 0 exactly). For i.i.d. patterns
    with per-pixel variance s2, DGI_N(x) -> s2 * (O(x) - <B>/<R>) and
    <B>/<R> -> spatial mean of O. Returns (raw DGI image, k = <B>/<R>).
    """
    if not 1 < n_shots <= patterns.shape[0]:
        raise ValueError(f"n_shots must be in (1, {patterns.shape[0]}], got {n_shots}")
    h, w = patterns.shape[1:]
    p = patterns[:n_shots].reshape(n_shots, -1).astype(np.float64)
    b = buckets[:n_shots].astype(np.float64)
    r = refs[:n_shots].astype(np.float64)
    k = float(b.mean() / r.mean())
    s = b - k * r                           # differential bucket signal
    dp = p - p.mean(axis=0, keepdims=True)
    dgi = (s @ dp) / n_shots
    return dgi.reshape(h, w), k


def normalize_dgi_calibrated(dgi: np.ndarray, pattern_var: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (unclipped estimate, clipped-to-[0,1] estimate) of O from DGI.

    O_hat(x) = DGI(x) / s2 + <B>/<R>. The division undoes the pattern variance
    (as for CGI); adding k = <B>/<R> restores the mean transmission that DGI
    removes. Both quantities come from the measurements, not the ground truth.
    """
    est = dgi / pattern_var + k
    return est, np.clip(est, 0.0, 1.0)


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-12:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


class StreamingDGI:
    """DGI for many objects at once, accumulated over pattern batches (no pattern storage).

    All objects see the same pattern sequence. After any number of shots N the
    accumulated sums give exactly the batch formula of dgi_reconstruct:

        DGI_N(x) = <S (P(x) - <P(x)>)>,   S = B - k R,  k = <B>/<R>
                 = [<B P(x)> - <B><P(x)>] - k [<R P(x)> - <R><P(x)>]

    (the second line follows because <S> = 0 exactly). Memory: O(M * D) for M
    objects of D pixels, independent of N.
    """

    def __init__(self, objects: np.ndarray):
        self.obj = objects.reshape(objects.shape[0], -1).astype(np.float64)   # (M, D)
        m, d = self.obj.shape
        self.n = 0
        self.sum_b = np.zeros(m)
        self.sum_r = 0.0
        self.sum_p = np.zeros(d)
        self.sum_bp = np.zeros((m, d))
        self.sum_rp = np.zeros(d)

    def update(self, pattern_batch: np.ndarray) -> np.ndarray:
        """Add a batch of patterns (b, H, W); returns its buckets (b, M)."""
        p = pattern_batch.reshape(pattern_batch.shape[0], -1).astype(np.float64)
        b = p @ self.obj.T                     # B_i for every object
        r = p.sum(axis=1)                      # R_i
        self.n += p.shape[0]
        self.sum_b += b.sum(axis=0)
        self.sum_r += r.sum()
        self.sum_p += p.sum(axis=0)
        self.sum_bp += b.T @ p
        self.sum_rp += r @ p
        return b

    def estimate(self, pattern_var: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (unclipped calibrated estimates (M, D), k (M,)) for the shots seen so far.

        O_hat = DGI / s2 + k, identical to normalize_dgi_calibrated.
        """
        n = self.n
        mean_b, mean_r, mean_p = self.sum_b / n, self.sum_r / n, self.sum_p / n
        k = mean_b / mean_r
        cov_bp = self.sum_bp / n - mean_b[:, None] * mean_p[None, :]
        cov_rp = self.sum_rp / n - mean_r * mean_p
        dgi = cov_bp - k[:, None] * cov_rp[None, :]
        return dgi / pattern_var + k[:, None], k
