"""Synthetic gravity scenes sampled at the real PSG station geometry.

A scene = regional field + target bodies (complex-scale intrusive cores = eruption-centre
zones, label 1) + distractor bodies (small plugs, elongated dykes/ridges, low-density bodies;
label 0) + white noise + an occasional gross station error.

All bodies are vertical cylinders (dykes = chains of touching cylinders with the density
scaled by 4/pi to conserve mass). Geometric parameters are drawn from discretised uniform
priors (100 m / 500 m steps) so that radial profiles can be cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cylinder import Profile

# ------------------------------------------------------------------ priors (documented)
PRIORS = dict(
    n_targets_p=[0.20, 0.35, 0.30, 0.15],          # P(0,1,2,3 targets)
    target_R=(1500, 4000), target_zt=(100, 1000), target_H=(1500, 3000), target_drho=(150.0, 400.0),
    n_plugs_mean=2.0, plug_R=(300, 1000), plug_zt=(100, 800), plug_H=(500, 2500), plug_drho=(150.0, 350.0),
    n_dykes_mean=1.0, dyke_L=(3000.0, 12000.0), dyke_a=(150, 750), dyke_zt=(100, 1000), dyke_H=(1000, 3000),
    dyke_drho=(100.0, 300.0),                       # sign random
    n_neg_mean=0.8, neg_R=(2000, 6000), neg_zt=(100, 300), neg_H=(500, 2000), neg_drho=(150.0, 400.0),
    plane_grad=(0.0, 2.0),                          # mGal/km
    grf_len=(4000.0, 12000.0), grf_std=(4.0, 12.0), # squared-exponential GRF (geological clutter)
    sigma=(1.0, 4.0), p_outlier=0.3, outlier_std=8.0,
    margin=2000.0,                                  # bodies may sit up to 2 km outside the hull bbox
)


def _qu(rng, lo, hi, step):
    """Uniform draw on the grid lo, lo+step, ..., hi."""
    return float(lo + step * rng.integers(0, int(round((hi - lo) / step)) + 1))


def _qlogu(rng, lo, hi, step):
    v = np.exp(rng.uniform(np.log(lo), np.log(hi)))
    return float(np.clip(step * round(v / step), lo, hi))


class ProfileBank:
    """Cache of radial profiles keyed by (R, zt, H); optionally persisted to .npz."""

    def __init__(self, path=None):
        self.cache = {}
        self.path = path
        if path is not None:
            try:
                z = np.load(path)
                for k, rr, gg in zip(z["keys"], z["r"], z["g"]):
                    p = Profile.__new__(Profile)
                    p.R, p.zt, p.zb = float(k[0]), float(k[1]), float(k[1] + k[2])
                    p.r, p.g = rr.astype(float), gg.astype(float)
                    self.cache[tuple(float(v) for v in k)] = p
            except FileNotFoundError:
                pass

    def save(self):
        if self.path is None or not self.cache:
            return
        keys = np.array(list(self.cache.keys()))
        n = max(len(p.r) for p in self.cache.values())
        r = np.full((len(keys), n), np.nan)
        g = np.full((len(keys), n), np.nan)
        for i, p in enumerate(self.cache.values()):
            r[i, :len(p.r)], g[i, :len(p.g)] = p.r, p.g
        np.savez_compressed(self.path, keys=keys, r=r, g=g)

    def __call__(self, R, zt, H):
        key = (float(R), float(zt), float(H))
        p = self.cache.get(key)
        if p is None:
            p = Profile(R, zt, zt + H)
            self.cache[key] = p
        return p


@dataclass
class Scene:
    d: np.ndarray                      # station values (mGal)
    targets: np.ndarray                # (k, 4): x, y, R, drho
    sigma: float
    parts: dict = field(default_factory=dict)


class SceneGenerator:
    def __init__(self, xy_st, hull, priors=None, bank=None):
        self.xy = np.asarray(xy_st, float)
        self.hull = hull
        self.pr = dict(PRIORS, **(priors or {}))
        self.bank = bank or ProfileBank()
        m = self.pr["margin"]
        self.lo = hull.min(0) - m
        self.hi = hull.max(0) + m
        self.c0 = self.xy.mean(0)

    def _dist(self, x, y):
        return np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)

    def _xy(self, rng):
        return rng.uniform(self.lo, self.hi)

    def sample(self, rng, keep_parts=False):
        pr = self.pr
        n = len(self.xy)
        parts = {}
        # regional: plane + squared-exponential Gaussian random field (random Fourier features)
        g = rng.uniform(*pr["plane_grad"]) / 1000.0
        az = rng.uniform(0, 2 * np.pi)
        plane = g * ((self.xy[:, 0] - self.c0[0]) * np.cos(az) + (self.xy[:, 1] - self.c0[1]) * np.sin(az))
        L = rng.uniform(*pr["grf_len"])
        s = rng.uniform(*pr["grf_std"])
        M = 300
        w = rng.normal(0.0, 1.0 / L, size=(M, 2))
        b = rng.uniform(0, 2 * np.pi, M)
        grf = s * np.sqrt(2.0 / M) * np.cos(self.xy @ w.T + b).sum(1)
        parts["regional"] = plane + grf

        # targets
        k = rng.choice(len(pr["n_targets_p"]), p=pr["n_targets_p"])
        tg = []
        tries = 0
        while len(tg) < k and tries < 200:
            tries += 1
            x, y = self._xy(rng)
            R = _qlogu(rng, *pr["target_R"], 100)
            if any(np.hypot(x - t[0], y - t[1]) < R + t[2] + 500.0 for t in tg):
                continue
            zt = _qu(rng, *pr["target_zt"], 100)
            H = _qu(rng, *pr["target_H"], 500)
            dr = rng.uniform(*pr["target_drho"])
            tg.append((x, y, R, dr, zt, H))
        sig_t = np.zeros(n)
        for x, y, R, dr, zt, H in tg:
            sig_t += dr * self.bank(R, zt, H)(self._dist(x, y))
        parts["targets"] = sig_t

        # distractors
        sig_d = np.zeros(n)
        for _ in range(rng.poisson(pr["n_plugs_mean"])):
            x, y = self._xy(rng)
            p = self.bank(_qu(rng, *pr["plug_R"], 100), _qu(rng, *pr["plug_zt"], 100), _qu(rng, *pr["plug_H"], 500))
            sig_d += rng.uniform(*pr["plug_drho"]) * p(self._dist(x, y))
        for _ in range(rng.poisson(pr["n_dykes_mean"])):
            x, y = self._xy(rng)
            Ld = rng.uniform(*pr["dyke_L"])
            a = _qu(rng, *pr["dyke_a"], 150)
            th = rng.uniform(0, np.pi)
            p = self.bank(a, _qu(rng, *pr["dyke_zt"], 100), _qu(rng, *pr["dyke_H"], 500))
            dr = rng.choice([-1.0, 1.0]) * rng.uniform(*pr["dyke_drho"]) * 4.0 / np.pi
            m = max(2, int(round(Ld / (2 * a))))
            offs = (np.arange(m) - (m - 1) / 2) * 2 * a
            for o in offs:
                sig_d += dr * p(self._dist(x + o * np.cos(th), y + o * np.sin(th)))
        for _ in range(rng.poisson(pr["n_neg_mean"])):
            x, y = self._xy(rng)
            p = self.bank(_qu(rng, *pr["neg_R"], 100), _qu(rng, *pr["neg_zt"], 100), _qu(rng, *pr["neg_H"], 500))
            sig_d -= rng.uniform(*pr["neg_drho"]) * p(self._dist(x, y))
        parts["distractors"] = sig_d

        sigma = rng.uniform(*pr["sigma"])
        noise = sigma * rng.standard_normal(n)
        if rng.uniform() < pr["p_outlier"]:
            noise[rng.integers(n)] += pr["outlier_std"] * rng.standard_normal()
        parts["noise"] = noise

        d = parts["regional"] + sig_t + sig_d + noise
        targets = np.array([[t[0], t[1], t[2], t[3]] for t in tg]).reshape(-1, 4)
        return Scene(d=d, targets=targets, sigma=sigma, parts=parts if keep_parts else {})


def cell_labels(cells, targets):
    """1 where a cell lies inside the footprint of any target (r <= R)."""
    y = np.zeros(len(cells), dtype=bool)
    for x0, y0, R, _ in targets:
        y |= np.hypot(cells[:, 0] - x0, cells[:, 1] - y0) <= R
    return y
