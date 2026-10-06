"""Independent checks of the POD experiment (run after run_pod.py)."""
import numpy as np
import pandas as pd

from cylinder import Profile
from detectors import plane_projector, tps_weights
from geometry import convex_hull, grid_nodes, lattice_in_polygon, load_psg, pairwise_dist
from run_pod import ALPHA, H, SEED, STEP, colmax

df = pd.read_csv("results/pod_results.csv")
thr = pd.read_csv("results/null_thresholds.csv")
out = []

# 1. calibration via the signal runs: tiny bodies (R = 250 m) carry ~no signal, and their noise is
#    independent of the noise used to set thresholds -> exceedance should be ~ALPHA
tiny = df[df.R_m == 250]
for det in ("D1", "D2"):
    r = tiny[f"exceed_{det}"]
    out.append(f"[1] false-alarm rate {det} (R=250 m runs, n={len(r)} configs x 500): "
               f"mean {r.mean():.3f}, range {r.min():.3f}-{r.max():.3f} (target {ALPHA})")

# 2. fresh independent null simulations for selected configurations
xy_psg, _ = load_psg("data/gravity_requested_area.csv")
hull = convex_hull(xy_psg)
grid, _ = grid_nodes(hull, STEP)
geoms = {"PSG-43": xy_psg, "Grid 1.00 km": lattice_in_polygon(hull, 1000)}
rng = np.random.default_rng(SEED + 999)
for gname, xy in geoms.items():
    P = plane_projector(xy)
    W = tps_weights(xy, grid)
    Nn = rng.standard_normal((len(xy), 4000))
    _, m2 = colmax(W, P @ Nn)
    t2 = thr[(thr.geometry == gname)].thr_D2_sigma.iloc[0]
    out.append(f"[2] {gname:12s} D2 fresh-null false-alarm = {(m2 > t2).mean():.3f}")
    D = pairwise_dist(grid, xy)
    for R, zt in [(1000, 100), (2000, 500)]:
        prof = Profile(R, zt, zt + H)
        S = prof(D)
        SP = S @ P
        sq = np.sqrt(np.einsum("ij,ij->i", SP, S))
        _, m1 = colmax(SP, Nn, sq)
        t1 = thr[(thr.geometry == gname) & (thr.R_m == R) & (thr.zt_m == zt)].thr_D1.iloc[0]
        out.append(f"[2] {gname:12s} D1 R={R} zt={zt} fresh-null false-alarm = {(m1 > t1).mean():.3f}")

# 3. monotonicity of POD in R (beyond 3 binomial standard errors)
bad = 0
for key, sub in df.groupby(["geometry", "zt_m", "drho", "sigma"]):
    sub = sub.sort_values("R_m")
    for det in ("pod_D1", "pod_D2"):
        p = sub[det].to_numpy()
        se = np.sqrt(np.maximum(p * (1 - p), 1e-4) / 500)
        drops = (p[:-1] - p[1:]) > 3 * np.maximum(se[:-1], se[1:])
        bad += drops.sum()
out.append(f"[3] POD decreases with R beyond 3 SE: {bad} cases")

# 4. POD increases with station density (fixed body) - fraction of pairs that violate beyond 3 SE
order = ["Grid 2.83 km", "Grid 2.00 km", "Grid 1.41 km", "Grid 1.00 km", "Grid 0.71 km", "Grid 0.50 km"]
viol = tot = 0
for key, sub in df[df.geometry.isin(order)].groupby(["R_m", "zt_m", "drho", "sigma"]):
    p = sub.set_index("geometry").loc[order, "pod_D1"].to_numpy()
    se = np.sqrt(np.maximum(p * (1 - p), 1e-4) / 500)
    viol += ((p[:-1] - p[1:]) > 3 * np.maximum(se[:-1], se[1:])).sum()
    tot += len(p) - 1
out.append(f"[4] D1 POD decreases with denser grid beyond 3 SE: {viol} of {tot} steps")

# 5. localisation error of D1 detections, PSG-43
s = df[(df.geometry == "PSG-43") & (df.drho == 300) & (df.sigma == 2.0) & (df.zt_m == 100)]
out.append("[5] PSG-43 D1 median location error (m) by R: " +
           ", ".join(f"{int(r)}: {e:.0f}" for r, e in zip(s.R_m, s.locerr_med_D1) if np.isfinite(e)))
print("\n".join(out))
