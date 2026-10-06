"""Linear operators used by the two detectors.

D1  Matched-filter scan (generalised likelihood ratio for a positive-amplitude body of
    known shape at unknown position, white noise, planar regional removed). Using the
    TRUE body shape makes D1 an optimistic upper bound on what any method can detect
    with the given station geometry and noise.
D2  Conventional map reading: exact thin-plate-spline interpolation of the
    plane-removed station values onto the scan grid, then the highest grid value.
"""
from __future__ import annotations

import numpy as np


def plane_projector(xy):
    """P = I - X (X'X)^-1 X', X = [1, x, y]; removes a best-fitting plane."""
    c = xy.mean(0)
    X = np.column_stack([np.ones(len(xy)), (xy - c) / 1e4])
    Q, _ = np.linalg.qr(X)
    return np.eye(len(xy)) - Q @ Q.T


def _tps_kernel(r_km):
    with np.errstate(divide="ignore", invalid="ignore"):
        k = r_km * r_km * np.log(r_km)
    return np.where(r_km > 0, k, 0.0)


def tps_weights(xy_st, xy_grid):
    """W (n_grid x n_st) such that grid = W @ d is the exact TPS interpolant of d."""
    s = xy_st / 1e3
    g = xy_grid / 1e3
    n = len(s)
    K = _tps_kernel(np.sqrt(((s[:, None] - s[None]) ** 2).sum(-1)))
    Pm = np.column_stack([np.ones(n), s - s.mean(0)])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K
    A[:n, n:] = Pm
    A[n:, :n] = Pm.T
    rhs = np.zeros((n + 3, n))
    rhs[:n] = np.eye(n)
    C = np.linalg.solve(A, rhs)
    Kg = _tps_kernel(np.sqrt(((g[:, None] - s[None]) ** 2).sum(-1)))
    Pg = np.column_stack([np.ones(len(g)), g - s.mean(0)])
    return np.hstack([Kg, Pg]) @ C
