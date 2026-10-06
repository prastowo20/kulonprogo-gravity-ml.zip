"""Emulation of a global gravity model (GGMplus-like) regional field for synthetic scenes,
and GGM-derived features.

Only the measured part of GGMplus (wavelengths >~10 km, from GOCE/GRACE/EGM2008) carries
information on subsurface density; its shorter wavelengths are topography-implied (RTM) and
cancel after a consistent complete Bouguer reduction. The emulated product is therefore
    G(x) = lowpass_10km(true Bouguer field)(x) + e(x),
with e a smooth Gaussian random error field (std s_err, length scale L_err).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from detectors import tps_weights

LAMBDA_C = 10_000.0                         # cut-off wavelength (m); Gaussian amplitude response 0.5
SIGMA_G = LAMBDA_C * np.sqrt(2 * np.log(2)) / (2 * np.pi)   # Gaussian filter std in space (m)
GGM_FEATURES = ["ggm_val", "ggm_grad", "ggm_lap", "res_tps", "res_nn3", "res_ring01"]


class GGMGrid:
    def __init__(self, fe, step=1000.0, margin=10_000.0):
        (x0, y0), (x1, y1) = fe.hull.min(0) - margin, fe.hull.max(0) + margin
        self.xs = np.arange(np.floor(x0 / step) * step, x1 + step, step)
        self.ys = np.arange(np.floor(y0 / step) * step, y1 + step, step)
        X, Y = np.meshgrid(self.xs, self.ys)
        self.nodes = np.column_stack([X.ravel(), Y.ravel()])
        self.shape = X.shape
        self.step = step
        ny, nx = self.shape
        self.pad = 16
        ky = 2 * np.pi * np.fft.fftfreq(ny + 2 * self.pad, d=step)
        kx = 2 * np.pi * np.fft.fftfreq(nx + 2 * self.pad, d=step)
        K2 = np.add.outer(ky ** 2, kx ** 2)
        self.H = np.exp(-0.5 * K2 * SIGMA_G ** 2)[:, :, None]
        self.fe = fe
        self.cells = fe.cells
        self.stations = fe.xy
        self.W_res = tps_weights(fe.xy, fe.cells)          # TPS of residual at cells (exact)
        D = np.hypot(fe.cells[:, None, 0] - fe.xy[None, :, 0], fe.cells[:, None, 1] - fe.xy[None, :, 1])
        w = np.where(D < 3000.0, 1.0 / np.maximum(D, 250.0), 0.0)
        s = w.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.W_ring01 = np.where(s > 0, w / s, np.nan)
        self.nn = fe.nn
        self.Wc = self._bilinear(fe.cells)
        self.Ws = self._bilinear(fe.xy)

    def _bilinear(self, pts):
        """Sparse (n_pts, n_nodes) bilinear interpolation matrix on the regular grid."""
        ny, nx = self.shape
        fx = (pts[:, 0] - self.xs[0]) / self.step
        fy = (pts[:, 1] - self.ys[0]) / self.step
        i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        tx, ty = fx - i0, fy - j0
        rows = np.repeat(np.arange(len(pts)), 4)
        cols = np.stack([j0 * nx + i0, j0 * nx + i0 + 1, (j0 + 1) * nx + i0, (j0 + 1) * nx + i0 + 1], 1).ravel()
        vals = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1).ravel()
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(pts), ny * nx))

    def lowpass(self, F, lambda_c=None):
        """F: (n_nodes, B) -> Gaussian low-pass (n_nodes, B); cut-off wavelength lambda_c (amplitude 0.5)
        (default LAMBDA_C = 10 km, as used for datasets.py)."""
        ny, nx = self.shape
        G = F.reshape(ny, nx, -1)
        p = self.pad
        if lambda_c is None or lambda_c == LAMBDA_C:
            H = self.H
        else:
            sig = lambda_c * np.sqrt(2 * np.log(2)) / (2 * np.pi)
            ky = 2 * np.pi * np.fft.fftfreq(ny + 2 * p, d=self.step)
            kx = 2 * np.pi * np.fft.fftfreq(nx + 2 * p, d=self.step)
            H = np.exp(-0.5 * np.add.outer(ky ** 2, kx ** 2) * sig ** 2)[:, :, None]
        Gp = np.pad(G, ((p, p), (p, p), (0, 0)), mode="reflect")
        out = np.real(np.fft.ifft2(np.fft.fft2(Gp, axes=(0, 1)) * H, axes=(0, 1)))[p:-p, p:-p]
        return out.reshape(ny * nx, -1)

    def error_field(self, rng, s_err, L_err=12_000.0, M=300):
        if s_err == 0:
            return np.zeros(len(self.nodes))
        w = rng.normal(0.0, 1.0 / L_err, size=(M, 2))
        b = rng.uniform(0, 2 * np.pi, M)
        return s_err * np.sqrt(2.0 / M) * np.cos(self.nodes @ w.T + b).sum(1)

    def features(self, G, D):
        """G: (n_nodes, B) emulated GGM grids; D: (n_st, B) station data -> (B, n_in, 6) float32."""
        ny, nx = self.shape
        h = self.step / 1e3
        Gg = G.reshape(ny, nx, -1)
        gy, gx = np.gradient(Gg, h, axis=(0, 1))
        grad = np.hypot(gx, gy).reshape(ny * nx, -1)
        lap = (np.gradient(gx, h, axis=1) + np.gradient(gy, h, axis=0)).reshape(ny * nx, -1)
        gc = self.Wc @ G
        gc = gc - gc.mean(0, keepdims=True)
        gs = self.Ws @ G
        r = D - gs
        r = r - r.mean(0, keepdims=True)                    # GGM-detrended station residuals
        feats = [gc, self.Wc @ grad, self.Wc @ lap, self.W_res @ r,
                 r[self.nn].mean(1), self.W_ring01 @ r]
        X = np.stack(feats, axis=-1)                        # (n_in, B, 6)
        return np.ascontiguousarray(X.transpose(1, 0, 2)).astype(np.float32)
