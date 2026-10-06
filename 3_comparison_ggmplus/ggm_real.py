"""Re-reduction of the GGMplus grid (G obs, z) to a complete Bouguer anomaly and band-wise
comparison with the PSG terrestrial CBA (research question 1, first pass).

Reduction: FAA = g - gamma(phi) + second-order free-air term; CBA = FAA - T(P), where T(P) is
the attraction of all topographic masses between sea level and the GGMplus surface (rho =
2670 kg/m^3, the RTM density of GGMplus), computed with right prisms: full 220 m resolution
within ~3 km (by 5 x 5 blocks), block-averaged 1.1 km prisms beyond, to the grid edge.

Usage: python ggm_real.py <path to 'Data GGMPlus Input Oasis.xls'>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import xlrd
from pyproj import Transformer

from geometry import convex_hull, inside, load_psg
from prism import G, gz_prisms

RHO = 2670.0
BLOCK = 5                    # 5 x 5 nodes = ~1.1 km output/aggregation blocks
NEAR_KM = 3.0
LAMBDA_C = 10_000.0
SIGMA_G = LAMBDA_C * np.sqrt(2 * np.log(2)) / (2 * np.pi)


def load_grid(path):
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    a = np.array([sh.row_values(r) for r in range(1, sh.nrows)], float)
    n = int(round(np.sqrt(len(a))))
    x, y, z, g = (a[:, k].reshape(n, n) for k in range(4))
    assert np.all(np.diff(x[0]) > 0) and np.all(np.diff(y[:, 0]) > 0), "unexpected grid ordering"
    return x, y, np.clip(z, 0, None), g


def normal_gravity(lat_deg):
    s2 = np.sin(np.radians(lat_deg)) ** 2
    return 978032.67715 * (1 + 0.001931851353 * s2) / np.sqrt(1 - 0.0066943800229 * s2)


def reduce_grid(x, y, z, g):
    lon, lat = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True).transform(x, y)
    s2 = np.sin(np.radians(lat)) ** 2
    faa = g - normal_gravity(lat) + (0.3087691 - 0.0004398 * s2) * z - 7.2125e-8 * z ** 2
    n = x.shape[0]
    dx = np.median(np.diff(x, axis=1))
    dy = np.median(np.diff(y, axis=0))
    nb = n // BLOCK
    # full-resolution prisms
    Pf = np.column_stack([(x - dx / 2).ravel(), (x + dx / 2).ravel(), (y - dy / 2).ravel(), (y + dy / 2).ravel(),
                          np.zeros(n * n), z.ravel()])
    bid = ((np.arange(n)[:, None] // BLOCK) * nb + (np.arange(n)[None, :] // BLOCK)).ravel()
    # block prisms (mean height)
    bx = x[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    by = y[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    bz = z[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK).mean((1, 3)).ravel()
    Pb = np.column_stack([bx - BLOCK * dx / 2, bx + BLOCK * dx / 2, by - BLOCK * dy / 2, by + BLOCK * dy / 2,
                          np.zeros(nb * nb), bz])
    # output points: node nearest each block centre
    oi = (np.arange(nb) * BLOCK + BLOCK // 2)
    I, J = np.meshgrid(oi, oi, indexing="ij")
    ox, oy, oz = x[I, J].ravel(), y[I, J].ravel(), z[I, J].ravel()
    T = np.zeros(len(ox))
    for k in range(len(ox)):
        d = np.hypot(bx - ox[k], by - oy[k]) / 1e3
        near_blocks = np.where(d <= NEAR_KM)[0]
        far_blocks = np.where(d > NEAR_KM)[0]
        near_nodes = np.isin(bid, near_blocks)
        obs = [[ox[k], oy[k], oz[k] + 0.1]]          # 0.1 m above the surface
        T[k] = gz_prisms(obs, Pf[near_nodes], RHO)[0] + gz_prisms(obs, Pb[far_blocks], RHO)[0]
    faa_o = faa[I, J].ravel()
    cba = faa_o - T
    sba = faa_o - 2 * np.pi * G * RHO * oz * 1e5
    return dict(x=ox, y=oy, z=oz, faa=faa_o, T=T, sba=sba, cba=cba, tc=cba - sba, shape=(nb, nb),
                step=float(BLOCK * (dx + dy) / 2))


def gauss_lowpass(values, shape, step):
    ny, nx = shape
    Gr = values.reshape(ny, nx)
    pad = 16
    Gp = np.pad(Gr, pad, mode="reflect")
    ky = 2 * np.pi * np.fft.fftfreq(Gp.shape[0], d=step)
    kx = 2 * np.pi * np.fft.fftfreq(Gp.shape[1], d=step)
    H = np.exp(-0.5 * np.add.outer(ky ** 2, kx ** 2) * SIGMA_G ** 2)
    return np.real(np.fft.ifft2(np.fft.fft2(Gp) * H))[pad:-pad, pad:-pad].ravel()


def at_points(grid, px, py, qx, qy):
    """Bilinear interpolation from a (nearly) regular grid given by node coords px, py."""
    from scipy.interpolate import griddata
    return griddata(np.column_stack([px, py]), grid, np.column_stack([qx, qy]), method="linear")


def poly2(xy, v, xy_eval=None):
    c = xy.mean(0)
    def A(p):
        u, w = (p[:, 0] - c[0]) / 1e4, (p[:, 1] - c[1]) / 1e4
        return np.column_stack([np.ones(len(p)), u, w, u * u, u * w, w * w])
    coef, *_ = np.linalg.lstsq(A(xy), v, rcond=None)
    return A(xy if xy_eval is None else xy_eval) @ coef, coef


def compare(red, xy_st, cba_psg):
    step = red["step"]                                  # output-block spacing (~1.1 km)
    out = {}
    lp = gauss_lowpass(red["cba"], red["shape"], step)
    fields = {"GGM CBA (all wavelengths)": red["cba"], "GGM CBA low-pass 10 km": lp,
              "GGM SBA (no terrain corr.)": red["sba"]}
    for k, f in fields.items():
        v = at_points(f, red["x"], red["y"], xy_st[:, 0], xy_st[:, 1])
        d = cba_psg - v
        out[k] = dict(r=float(np.corrcoef(cba_psg, v)[0, 1]), bias=float(d.mean()), sd_diff=float(d.std()),
                      rmse=float(np.sqrt((d ** 2).mean())))
    # PSG kernel-smoothed at the same scale, vs GGM low-pass
    D = np.hypot(xy_st[:, None, 0] - xy_st[None, :, 0], xy_st[:, None, 1] - xy_st[None, :, 1])
    W = np.exp(-0.5 * (D / SIGMA_G) ** 2)
    psg_lp = (W @ cba_psg) / W.sum(1)
    v_lp = at_points(lp, red["x"], red["y"], xy_st[:, 0], xy_st[:, 1])
    d = psg_lp - v_lp
    out["low-pass both (10 km)"] = dict(r=float(np.corrcoef(psg_lp, v_lp)[0, 1]), bias=float(d.mean()),
                                        sd_diff=float(d.std()), rmse=float(np.sqrt((d ** 2).mean())))
    # trend (2nd order) vs residual bands
    v_all = at_points(red["cba"], red["x"], red["y"], xy_st[:, 0], xy_st[:, 1])
    tp, cp = poly2(xy_st, cba_psg)
    tg, cg = poly2(xy_st, v_all)
    out["2nd-order trend"] = dict(r=float(np.corrcoef(tp, tg)[0, 1]), sd_diff=float((tp - tg - (tp - tg).mean()).std()),
                                  psg_trend_sd=float(tp.std()), ggm_trend_sd=float(tg.std()))
    rp, rg = cba_psg - tp, v_all - tg
    out["residual after trend"] = dict(r=float(np.corrcoef(rp, rg)[0, 1]), psg_sd=float(rp.std()),
                                       ggm_sd=float(rg.std()))
    out["implied_ggm_error_sd_lowpass_mGal"] = float(d.std())
    return out, lp


def main(path):
    x, y, z, g = load_grid(path)
    red = reduce_grid(x, y, z, g)
    xy_st, df = load_psg("data/gravity_requested_area.csv")
    res, lp = compare(red, xy_st, df.CBA.to_numpy())
    np.savez_compressed("results/ggm_real_reduced.npz", **{k: v for k, v in red.items() if k not in ("shape", "step")},
                        shape=np.array(red["shape"]), step=red["step"], cba_lowpass=lp)
    stats = dict(tc_min=float(red["tc"].min()), tc_median=float(np.median(red["tc"])), tc_max=float(red["tc"].max()),
                 corr_cba_z=float(np.corrcoef(red["cba"], red["z"])[0, 1]),
                 corr_sba_z=float(np.corrcoef(red["sba"], red["z"])[0, 1]),
                 corr_faa_z=float(np.corrcoef(red["faa"], red["z"])[0, 1]))
    json.dump(dict(stats=stats, comparison=res), open("results/ggm_real_comparison.json", "w"), indent=1)
    print(json.dumps(dict(stats=stats, comparison=res), indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
