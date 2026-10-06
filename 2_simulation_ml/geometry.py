"""Survey geometries, survey polygon and sampling helpers (UTM 49S, metres)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.path import Path


def load_psg(csv_path):
    df = pd.read_csv(csv_path)
    return df[["X", "Y"]].to_numpy(float), df


def convex_hull(xy):
    """Andrew's monotone chain; returns CCW hull vertices (k, 2)."""
    pts = sorted(map(tuple, np.asarray(xy, float)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside(poly, pts):
    return Path(poly).contains_points(np.asarray(pts, float))


def dist_to_boundary(poly, pts):
    """Minimum distance from each point to the polygon edges."""
    pts = np.asarray(pts, float)
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    proj = a[None] + t[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(-1)).min(axis=1)


def lattice_in_polygon(poly, spacing):
    """Regular square lattice clipped to the polygon (origin at bbox corner + spacing/2)."""
    (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
    xs = np.arange(x0 + spacing / 2, x1, spacing)
    ys = np.arange(y0 + spacing / 2, y1, spacing)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    return pts[inside(poly, pts)]


def grid_nodes(poly, step):
    """Scan/map grid nodes inside the polygon; also returns the full grid axes."""
    (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
    xs = np.arange(np.floor(x0 / step) * step, x1 + step, step)
    ys = np.arange(np.floor(y0 / step) * step, y1 + step, step)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    m = inside(poly, pts)
    return pts[m], (xs, ys, m.reshape(X.shape))


def sample_inner(poly, buffer, n, rng):
    """Uniform random points inside the polygon at least `buffer` m from its edge."""
    (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
    out = []
    while sum(len(o) for o in out) < n:
        cand = np.column_stack([rng.uniform(x0, x1, 4 * n), rng.uniform(y0, y1, 4 * n)])
        cand = cand[inside(poly, cand)]
        cand = cand[dist_to_boundary(poly, cand) >= buffer]
        out.append(cand)
    return np.vstack(out)[:n]


def pairwise_dist(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.sqrt((a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2)
