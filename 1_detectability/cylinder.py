"""Vertical gravity attraction of a homogeneous vertical circular cylinder.

Used as the forward model for volcanic necks, plugs and intrusive stocks.

Conventions
-----------
* Observation points lie on the horizontal plane z = 0; depth z is positive downward.
* Cylinder: radius R, top depth zt > 0, bottom depth zb > zt (metres).
* Density contrast drho in kg/m^3. Output g_z in mGal (positive = excess attraction).

Method
------
The vertical integral of a vertical line element is exact:
    dg = G*drho*dA*[1/sqrt(s^2+zt^2) - 1/sqrt(s^2+zb^2)],
with s the horizontal distance between the element and the observation point.
Writing dA = s ds dtheta in polar coordinates centred on the observation point,
the radial integral is also closed form:
    int s*(...) ds = F(s) = sqrt(s^2+zt^2) - sqrt(s^2+zb^2),
so only a smooth 1-D angular integral remains:
* observation point inside the cylinder footprint (r < R): full circle, periodic
  trapezoid rule (spectrally accurate);
* outside (r >= R): wedge |phi| <= asin(R/r), substitution phi = phi0*sin(psi)
  removes the square-root endpoint behaviour; Gauss-Legendre in psi.
"""
from __future__ import annotations

import numpy as np

G = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
SI2MGAL = 1.0e5  # 1 m/s^2 = 1e5 mGal


def _F(s, zt, zb):
    return np.sqrt(s * s + zt * zt) - np.sqrt(s * s + zb * zb)


def gz_cylinder(r, R, zt, zb, drho, n_theta=2048, n_psi=256):
    """g_z (mGal) at horizontal distance(s) r (m) from the cylinder axis."""
    if not (zb > zt > 0 and R > 0):
        raise ValueError("require zb > zt > 0 and R > 0")
    r = np.atleast_1d(np.asarray(r, dtype=float))
    out = np.empty_like(r)

    inside = r < R
    if inside.any():
        th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        ri = r[inside][:, None]
        pu = ri * np.cos(th)[None, :]  # P.u with P = (r, 0)
        s2 = -pu + np.sqrt(pu * pu + R * R - ri * ri)
        val = _F(s2, zt, zb) - _F(0.0, zt, zb)
        out[inside] = val.mean(axis=1) * 2.0 * np.pi

    outside = ~inside
    if outside.any():
        ro = r[outside][:, None]
        phi0 = np.arcsin(np.clip(R / ro, 0.0, 1.0))
        x, w = np.polynomial.legendre.leggauss(n_psi)
        psi = 0.5 * np.pi * x
        wpsi = 0.5 * np.pi * w
        phi = phi0 * np.sin(psi)[None, :]
        jac = phi0 * np.cos(psi)[None, :]
        c = ro * np.cos(phi)
        disc = np.sqrt(np.clip(R * R - (ro * np.sin(phi)) ** 2, 0.0, None))
        val = (_F(c + disc, zt, zb) - _F(c - disc, zt, zb)) * jac
        out[outside] = (val * wpsi[None, :]).sum(axis=1)

    return G * drho * out * SI2MGAL


def gz_on_axis(R, zt, zb, drho):
    """Closed-form on-axis attraction (mGal)."""
    return 2.0 * np.pi * G * drho * ((zb - zt) + np.sqrt(R * R + zt * zt) - np.sqrt(R * R + zb * zb)) * SI2MGAL


def gz_line(r, R, zt, zb, drho):
    """Vertical line mass with the cylinder's cross-section (far-field limit), mGal."""
    r = np.asarray(r, dtype=float)
    return G * drho * np.pi * R * R * (1.0 / np.sqrt(r * r + zt * zt) - 1.0 / np.sqrt(r * r + zb * zb)) * SI2MGAL


def gz_bruteforce(r, R, zt, zb, drho, cell=5.0):
    """Independent check: disk cross-section cut into square cells of side `cell` (m),
    each treated as a vertical line element (exact in z). Slow; tests only."""
    n = int(np.ceil(R / cell))
    xs = (np.arange(-n, n) + 0.5) * cell
    X, Y = np.meshgrid(xs, xs)
    m = X * X + Y * Y <= R * R
    X, Y = X[m], Y[m]
    dA = cell * cell
    out = []
    for ri in np.atleast_1d(r):
        s2 = (X - ri) ** 2 + Y * Y
        out.append(np.sum(1.0 / np.sqrt(s2 + zt * zt) - 1.0 / np.sqrt(s2 + zb * zb)) * dA)
    return G * drho * np.array(out) * SI2MGAL


class Profile:
    """Tabulated radial profile g(r) for fast evaluation at many distances.

    Built once per (R, zt, zb) for drho = 1 kg/m^3; multiply by drho when used.
    """

    def __init__(self, R, zt, zb, rmax=60_000.0):
        r_near = np.linspace(0.0, 4.0 * R + 4.0 * zb, 1601)
        r_far = np.geomspace(r_near[-1] * 1.001, rmax, 400)
        self.r = np.concatenate([r_near, r_far])
        self.g = gz_cylinder(self.r, R, zt, zb, 1.0)
        self.R, self.zt, self.zb = R, zt, zb

    def __call__(self, dist):
        d = np.asarray(dist, dtype=float)
        g = np.interp(d, self.r, self.g, right=np.nan)
        far = np.isnan(g)
        if far.any():  # beyond table: line-mass limit (negligible amplitude there)
            g[far] = gz_line(d[far], self.R, self.zt, self.zb, 1.0)
        return g
