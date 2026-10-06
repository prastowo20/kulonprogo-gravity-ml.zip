"""Reduction-density consistency between the PSG CBA and the re-reduced GGMplus CBA (RQ1).

The PSG file gives CBA only (no elevation, no g_obs, no terrain correction, reduction density
not stated). The GGMplus CBA was re-reduced with 2670 kg/m^3 (the RTM density of GGMplus). Two
hypotheses for the PSG reduction density are tested:
    H1: PSG reduced with 2670 kg/m^3  -> use the PSG CBA as is;
    H2: PSG reduced with 2840 kg/m^3  -> convert to 2670 kg/m^3:
        CBA_2670 = CBA_2840 + T_2670(P) * (2840 - 2670) / 2670,
where T_2670(P) is the attraction of all topographic masses (sea level to surface) at the
station, computed with the same prism model as the GGMplus reduction (station height taken
from the GGMplus/SRTM grid node). Both hypotheses are compared with GGMplus on the same bands.

Usage: python psg_density.py data/ggmplus_oasis.xls
Writes: results/psg_density.json and data/derived/psg_cba_H1_H2.csv (PSG-derived, not public).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from geometry import load_psg
from ggm_real import BLOCK, NEAR_KM, RHO, compare, gauss_lowpass, at_points, poly2
from prism import gz_prisms

LAMBDA_EGM = 18_300.0        # full wavelength of EGM2008 degree 2190 (360 deg / 2190 * 111.2 km)


def topo_attraction_at(px, py, x, y, z):
    """T_2670 at arbitrary points, placed 0.1 m above the nearest grid node (as in reduce_grid)."""
    n = x.shape[0]
    dx = np.median(np.diff(x, axis=1))
    dy = np.median(np.diff(y, axis=0))
    nb = n // BLOCK
    Pf = np.column_stack([(x - dx / 2).ravel(), (x + dx / 2).ravel(), (y - dy / 2).ravel(), (y + dy / 2).ravel(),
                          np.zeros(n * n), z.ravel()])
    bid = ((np.arange(n)[:, None] // BLOCK) * nb + (np.arange(n)[None, :] // BLOCK)).ravel()
    bx = x[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    by = y[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    bz = z[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    Pb = np.column_stack([bx - BLOCK * dx / 2, bx + BLOCK * dx / 2, by - BLOCK * dy / 2, by + BLOCK * dy / 2,
                          np.zeros(nb * nb), bz])
    xr, yr, zr = x.ravel(), y.ravel(), z.ravel()
    T, H = np.zeros(len(px)), np.zeros(len(px))
    for k in range(len(px)):
        i = np.argmin(np.hypot(xr - px[k], yr - py[k]))
        H[k] = zr[i]
        d = np.hypot(bx - px[k], by - py[k]) / 1e3
        near = np.isin(bid, np.where(d <= NEAR_KM)[0])
        far = d > NEAR_KM
        obs = [[px[k], py[k], zr[i] + 0.1]]
        T[k] = gz_prisms(obs, Pf[near], RHO)[0] + gz_prisms(obs, Pb[far], RHO)[0]
    return T, H


def band_stats(red, xy, cba, lam):
    """Both fields low-passed at wavelength lam (GGM: FFT Gaussian on its grid; PSG: Gaussian kernel)."""
    sig = lam * np.sqrt(2 * np.log(2)) / (2 * np.pi)
    import ggm_real
    old = ggm_real.SIGMA_G
    ggm_real.SIGMA_G = sig
    try:
        lp = gauss_lowpass(red["cba"], tuple(red["shape"]), float(red["step"]))
    finally:
        ggm_real.SIGMA_G = old
    D = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    W = np.exp(-0.5 * (D / sig) ** 2)
    p = (W @ cba) / W.sum(1)
    v = at_points(lp, red["x"], red["y"], xy[:, 0], xy[:, 1])
    d = p - v
    return dict(r=float(np.corrcoef(p, v)[0, 1]), sd_diff=float(d.std()), psg_sd=float(p.std()), ggm_sd=float(v.std()))


def main(path):
    from ggm_real import load_grid
    x, y, z, g = load_grid(path)
    xy, df = load_psg("data/gravity_requested_area.csv")
    T, H = topo_attraction_at(xy[:, 0], xy[:, 1], x, y, z)
    cba1 = df.CBA.to_numpy()
    cba2 = cba1 + T * (2840.0 - 2670.0) / 2670.0
    z_red = np.load("results/ggm_real_reduced.npz")
    red = {k: z_red[k] for k in z_red.files}
    out = dict(station_height_from_grid_m=dict(min=float(H.min()), median=float(np.median(H)), max=float(H.max())),
               T2670_at_stations_mGal=dict(min=float(T.min()), median=float(np.median(T)), max=float(T.max())),
               H2_minus_H1_mGal=dict(min=float((cba2 - cba1).min()), max=float((cba2 - cba1).max()),
                                     sd=float((cba2 - cba1).std())))
    for name, c in [("H1_psg_2670", cba1), ("H2_psg_2840_converted", cba2)]:
        res, _ = compare(red, xy, c)
        res["band_lowpass_10km"] = band_stats(red, xy, c, 10_000.0)
        res["band_lowpass_EGM2008_limit_18.3km"] = band_stats(red, xy, c, LAMBDA_EGM)
        res["corr_psg_cba_with_station_height"] = float(np.corrcoef(c, H)[0, 1])
        out[name] = res
    os.makedirs("data/derived", exist_ok=True)
    pd.DataFrame(dict(X=xy[:, 0], Y=xy[:, 1], h_grid_m=H, T2670_mGal=T, CBA_H1=cba1, CBA_H2_2670=cba2)).to_csv(
        "data/derived/psg_cba_H1_H2.csv", index=False)
    json.dump(out, open("results/psg_density.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
