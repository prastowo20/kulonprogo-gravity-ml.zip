"""Monte Carlo probability-of-detection (POD) experiment for eroded eruption centres.

Question: with the PSG terrestrial gravity geometry (43 stations), how large must an
isolated dense intrusive body (neck / plug / stock, vertical cylinder) be before it can be
detected AND located? How does this change with station spacing?

Usage:  python run_pod.py            (full run)
        python run_pod.py --quick    (smoke test)
"""
from __future__ import annotations

import json
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

from cylinder import Profile
from detectors import plane_projector, tps_weights
from geometry import (convex_hull, dist_to_boundary, grid_nodes, lattice_in_polygon, load_psg,
                      pairwise_dist, polygon_area, sample_inner)

SEED = 20260923
R_LIST = [250, 500, 750, 1000, 1500, 2000, 3000, 4000]   # cylinder radius (m)
ZT_LIST = [100, 500, 1000]                                  # depth to top (m)
H = 2000.0                                                  # vertical extent (m)
DRHO_LIST = [200.0, 300.0]                                  # density contrast (kg/m^3)
SIGMA_LIST = [2.0, 5.5]                                     # white noise (mGal)
SPACINGS = [2830, 2000, 1410, 1000, 710, 500]               # regular-lattice spacings (m)
N_BODY, N_NULL = 500, 4000
ALPHA = 0.05            # family-wise false-alarm rate over the whole survey polygon
BUFFER = 2000.0         # bodies at least this far inside the station hull (m)
STEP = 250.0            # scan / map grid step (m)
CHUNK = 500

if "--quick" in sys.argv:
    R_LIST, ZT_LIST, DRHO_LIST, SPACINGS = [500, 2000], [100], [300.0], [1000]
    N_BODY, N_NULL = 60, 300


def chunks(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def colmax(A, B, scale=None):
    """max over rows of (A @ B) [/ scale], chunked over columns of B; returns (argmax, max)."""
    idx = np.empty(B.shape[1], dtype=int)
    val = np.empty(B.shape[1])
    for sl in chunks(B.shape[1], CHUNK):
        M = A @ B[:, sl]
        if scale is not None:
            M /= scale[:, None]
        i = M.argmax(0)
        idx[sl] = i
        val[sl] = M[i, np.arange(M.shape[1])]
    return idx, val


def main():
    t_start = time.time()
    xy_psg, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy_psg)
    grid, _ = grid_nodes(hull, STEP)
    rng = np.random.default_rng(SEED)
    bodies = sample_inner(hull, BUFFER, N_BODY, rng)

    geoms = {"PSG-43": xy_psg}
    for s in SPACINGS:
        geoms[f"Grid {s / 1000:.2f} km"] = lattice_in_polygon(hull, s)

    # context numbers for the report
    d_near = pairwise_dist(bodies, xy_psg).min(1)
    ctx = dict(hull_area_km2=polygon_area(hull) / 1e6, n_grid=len(grid), n_bodies=N_BODY,
               body_to_nearest_station_m=dict(median=float(np.median(d_near)),
                                              p90=float(np.percentile(d_near, 90))),
               n_stations={k: int(len(v)) for k, v in geoms.items()})
    print(json.dumps(ctx, indent=1))

    rows, nulls = [], []
    for gi, (gname, xy) in enumerate(geoms.items()):
        t0 = time.time()
        n = len(xy)
        P = plane_projector(xy)
        W = tps_weights(xy, grid)
        grng = np.random.default_rng([SEED, gi])
        Nnull = grng.standard_normal((n, N_NULL))
        Nsig = grng.standard_normal((n, N_BODY))
        PNnull = P @ Nnull

        # D2 threshold (units of sigma; linear operator so independent of sigma)
        _, m2 = colmax(W, PNnull)
        thr2 = float(np.quantile(m2, 1 - ALPHA))

        Dcs = pairwise_dist(grid, xy)
        Dbs = pairwise_dist(bodies, xy)
        ar = np.arange(N_BODY)
        for R in R_LIST:
            tol = max(float(R), 1000.0)
            for zt in ZT_LIST:
                prof = Profile(float(R), float(zt), float(zt) + H)
                S = prof(Dcs)
                SP = S @ P
                sq = np.sqrt(np.einsum("ij,ij->i", SP, S))
                _, m1 = colmax(SP, Nnull, sq)
                thr1 = float(np.quantile(m1, 1 - ALPHA))
                nulls.append(dict(geometry=gname, R_m=R, zt_m=zt, thr_D1=thr1, thr_D2_sigma=thr2))
                sig = prof(Dbs).T                     # n x N_BODY, drho = 1 kg/m^3
                peak1 = float(prof(np.array([0.0]))[0])
                for drho in DRHO_LIST:
                    for sigma in SIGMA_LIST:
                        d = drho * sig + sigma * Nsig
                        i1, z1 = colmax(SP, d, sigma * sq)
                        e1 = np.hypot(*(grid[i1] - bodies).T)
                        det1 = (z1 > thr1) & (e1 <= tol)
                        i2, v2 = colmax(W, P @ d)
                        e2 = np.hypot(*(grid[i2] - bodies).T)
                        det2 = (v2 > thr2 * sigma) & (e2 <= tol)
                        rows.append(dict(
                            geometry=gname, n_stations=n, R_m=R, zt_m=zt, H_m=H, drho=drho, sigma=sigma,
                            peak_mgal=drho * peak1,
                            pod_D1=det1.mean(), pod_D2=det2.mean(),
                            exceed_D1=(z1 > thr1).mean(), exceed_D2=(v2 > thr2 * sigma).mean(),
                            locerr_med_D1=float(np.median(e1[det1])) if det1.any() else np.nan,
                            locerr_med_D2=float(np.median(e2[det2])) if det2.any() else np.nan,
                            mean_st_above_2sigma=float(((drho * sig) > 2 * sigma).sum(0).mean())))
        print(f"{gname:14s} n={n:5d}  thr_D2={thr2:5.2f} sigma  {time.time() - t0:6.1f} s", flush=True)

    tag = "_quick" if "--quick" in sys.argv else ""
    pd.DataFrame(rows).to_csv(f"results/pod_results{tag}.csv", index=False)
    pd.DataFrame(nulls).to_csv(f"results/null_thresholds{tag}.csv", index=False)
    meta = dict(seed=SEED, R_list=R_LIST, zt_list=ZT_LIST, H=H, drho_list=DRHO_LIST, sigma_list=SIGMA_LIST,
                spacings=SPACINGS, n_body=N_BODY, n_null=N_NULL, alpha=ALPHA, buffer=BUFFER, step=STEP,
                python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, matplotlib=matplotlib.__version__,
                runtime_s=round(time.time() - t_start, 1), context=ctx)
    with open(f"results/meta{tag}.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
