"""Vertical attraction of right rectangular prisms (Nagy et al., 2000, closed form).

Coordinates in metres, z positive UP. Observation points (xp, yp, zp); prisms with limits
x1<x2, y1<y2, z1<z2 (z1 = bottom, z2 = top) and density rho (kg/m^3). Returns g_z in mGal,
positive downward (attraction towards the mass below the point).
"""
from __future__ import annotations

import numpy as np

G = 6.67430e-11


def _kernel(x, y, z):
    r = np.sqrt(x * x + y * y + z * z)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(x != 0, x * np.log(np.where(y + r > 0, y + r, 1.0)), 0.0)
        t2 = np.where(y != 0, y * np.log(np.where(x + r > 0, x + r, 1.0)), 0.0)
        t3 = np.where(z != 0, z * np.arctan((x * y) / np.where(z != 0, z * r, 1.0)), 0.0)
    return t1 + t2 - t3


def gz_prisms(obs, prisms, rho, chunk=4_000_000):
    """obs: (n, 3); prisms: (m, 6) [x1, x2, y1, y2, z1, z2]; rho: scalar or (m,). -> (n,) mGal."""
    obs = np.asarray(obs, float)
    P = np.asarray(prisms, float)
    rho = np.broadcast_to(np.asarray(rho, float), (len(P),))
    out = np.zeros(len(obs))
    step = max(1, chunk // max(len(P), 1))
    for a in range(0, len(obs), step):
        o = obs[a:a + step]
        X = [P[None, :, 0] - o[:, None, 0], P[None, :, 1] - o[:, None, 0]]
        Y = [P[None, :, 2] - o[:, None, 1], P[None, :, 3] - o[:, None, 1]]
        Z = [P[None, :, 4] - o[:, None, 2], P[None, :, 5] - o[:, None, 2]]
        s = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = s + (-1) ** (i + j + k) * _kernel(X[i], Y[j], Z[k])
        # sign: with z up, mass below the point gives positive (downward) attraction
        out[a:a + step] = -(s * rho[None, :]).sum(1)
    return G * out * 1e5


if __name__ == "__main__":
    # validation
    rho = 2670.0
    # 1) infinite-slab limit: very wide prism, point on its top centre -> 2 pi G rho H
    H = 1000.0
    g = gz_prisms([[0, 0, 0]], [[-1e7, 1e7, -1e7, 1e7, -H, 0.0]], rho)[0]
    print("slab: %.4f mGal vs 2piGrhoH = %.4f" % (g, 2 * np.pi * G * rho * H * 1e5))
    # 2) far field vs point mass
    a = 100.0
    obs = np.array([[5000.0, 3000.0, 2000.0]])
    g = gz_prisms(obs, [[-a, a, -a, a, -a, a]], rho)[0]
    M = rho * (2 * a) ** 3
    r = np.linalg.norm(obs[0])
    print("point mass: %.6e vs %.6e mGal" % (g, G * M * obs[0, 2] / r ** 3 * 1e5))
    # 3) against the vertical cylinder model (square prism of equal area), off-axis
    import sys
    sys.path.insert(0, ".")
    from cylinder import gz_cylinder
    R, zt, zb = 1000.0, 100.0, 2100.0
    s = R * np.sqrt(np.pi) / 2
    for d in (0.0, 3000.0, 8000.0):
        gp = gz_prisms([[d, 0, 0]], [[-s, s, -s, s, -zb, -zt]], 300.0)[0]
        gc = gz_cylinder(d, R, zt, zb, 300.0)[0]
        print("cylinder vs equal-area prism at r=%5.0f m: %.4f vs %.4f mGal" % (d, gc, gp))
    # 4) point exactly on a prism top corner/edge (singular terms guarded)
    print("point on prism top face corner:", gz_prisms([[0, 0, 0]], [[0, 100, 0, 100, -50, 0]], rho))
