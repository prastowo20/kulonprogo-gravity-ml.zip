"""Location-free gravity features on a 500 m cell grid (batch-capable).

No coordinates are used as features. Geometry-only features (station counts / distances)
tell the model where the data are sparse; they carry no prior on where centres are because
training bodies are placed uniformly at random.
"""
from __future__ import annotations

import warnings

import numpy as np

from cylinder import Profile
from detectors import plane_projector, tps_weights
from geometry import inside, pairwise_dist

RINGS = [(0, 1500), (1500, 3000), (3000, 4500), (4500, 7000), (7000, 10000)]
ANNULI = [(3000, 6000), (6000, 10000)]
MF_RADII = (1500.0, 2500.0, 3500.0)
MF_ZT, MF_H = 300.0, 2000.0

FEATURES = (["tps", "thdr", "lap", "vdr", "tdr", "asig", "con_3_6", "con_6_10"]
            + [f"ring{i}_mean" for i in range(len(RINGS))]
            + [f"ring{i}_n" for i in range(len(RINGS))]
            + ["nn1_dist", "nn1_val", "nn3_val", "nn3_dist", "ring_in_minus_out"]
            + [f"mf_R{int(r / 100) / 10:g}" for r in MF_RADII])


class FeatureExtractor:
    def __init__(self, xy_st, hull, step=500.0):
        self.xy = np.asarray(xy_st, float)
        self.hull = hull
        self.step = step
        (x0, y0), (x1, y1) = hull.min(0), hull.max(0)
        xs = np.arange(np.floor(x0 / step) * step, x1 + step, step)
        ys = np.arange(np.floor(y0 / step) * step, y1 + step, step)
        X, Y = np.meshgrid(xs, ys)
        self.xs, self.ys, self.shape = xs, ys, X.shape
        self.cells_all = np.column_stack([X.ravel(), Y.ravel()])
        self.inside = inside(hull, self.cells_all)
        self.cells = self.cells_all[self.inside]
        n_in = len(self.cells)

        self.P = plane_projector(self.xy)
        self.W = tps_weights(self.xy, self.cells_all)                       # (n_all, n_st)
        D = pairwise_dist(self.cells, self.xy)                              # (n_in, n_st)

        self.ring_W, self.ring_n = [], []
        for lo, hi in RINGS:
            m = (D >= lo) & (D < hi)
            w = np.where(m, 1.0 / np.maximum(D, 250.0), 0.0)
            s = w.sum(1, keepdims=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.ring_W.append(np.where(s > 0, w / s, np.nan))
            self.ring_n.append(m.sum(1).astype(float))
        order = np.argsort(D, axis=1)[:, :3]
        self.nn = order
        self.nn_d = np.take_along_axis(D, order, 1) / 1e3

        Dc = pairwise_dist(self.cells, self.cells)
        self.ann = []
        for lo, hi in ANNULI:
            m = ((Dc >= lo) & (Dc < hi)).astype(np.float64)
            self.ann.append(m / np.maximum(m.sum(1, keepdims=True), 1.0))

        self.MF = []
        for R in MF_RADII:
            S = Profile(R, MF_ZT, MF_ZT + MF_H)(D)
            SP = S @ self.P
            self.MF.append(SP / np.sqrt(np.einsum("ij,ij->i", SP, S))[:, None])

        ny, nx = self.shape
        self.pad = 24
        ky = 2 * np.pi * np.fft.fftfreq(ny + 2 * self.pad, d=step / 1e3)
        kx = 2 * np.pi * np.fft.fftfreq(nx + 2 * self.pad, d=step / 1e3)
        self.K = np.hypot(*np.meshgrid(kx, ky))[:, :, None]                # rad/km
        self.n_in = n_in

    def residual(self, D):
        return self.P @ D

    def transform(self, D):
        """D: (n_st,) or (n_st, B) station values -> X: (B, n_in, n_features) float32."""
        D = np.asarray(D, float)
        if D.ndim == 1:
            D = D[:, None]
        B = D.shape[1]
        r = self.P @ D                                                      # plane removed
        ny, nx = self.shape
        grid = (self.W @ r).reshape(ny, nx, B)
        h = self.step / 1e3
        gy, gx = np.gradient(grid, h, axis=(0, 1))
        thdr = np.hypot(gx, gy)
        gyy = np.gradient(gy, h, axis=0)
        gxx = np.gradient(gx, h, axis=1)
        lap = gxx + gyy
        p = self.pad
        gp = np.pad(grid, ((p, p), (p, p), (0, 0)), mode="reflect")
        vdr = np.real(np.fft.ifft2(np.fft.fft2(gp, axes=(0, 1)) * self.K, axes=(0, 1)))[p:-p, p:-p]
        tdr = np.arctan2(vdr, thdr)
        asig = np.hypot(thdr, vdr)
        ins = self.inside.reshape(ny, nx)
        take = lambda a: a[ins]                                             # (n_in, B)
        m = take(grid)
        feats = [m, take(thdr), take(lap), take(vdr), take(tdr), take(asig)]
        feats += [m - A @ m for A in self.ann]
        rings = [Wk @ r for Wk in self.ring_W]
        feats += rings
        feats += [np.repeat(n[:, None], B, 1) for n in self.ring_n]
        nnv = r[self.nn]                                                    # (n_in, 3, B)
        feats += [np.repeat(self.nn_d[:, :1], B, 1), nnv[:, 0], nnv.mean(1),
                  np.repeat(self.nn_d.mean(1, keepdims=True), B, 1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            inner = np.nanmean(np.stack(rings[:2]), axis=0)
        feats.append(inner - rings[3])
        feats += [M @ r for M in self.MF]
        X = np.stack(feats, axis=-1)                                        # (n_in, B, F)
        return np.ascontiguousarray(X.transpose(1, 0, 2)).astype(np.float32)
