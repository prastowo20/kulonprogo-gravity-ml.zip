"""Object-level (free-response) evaluation of the saved test predictions with non-maximum
suppression over +/-1.5 km (7 x 7 cells), so a single broad high counts as one peak."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import SCENARIOS
from train import object_eval, recall_at_fp

WIN = 7


def main():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    frocs, summ = [], []
    for name in SCENARIOS:
        z = np.load(f"results/test_{name}.npz", allow_pickle=True)
        P = z["p"].reshape(-1, fe.n_in)
        tg, froc = object_eval(fe, P, list(z["targets"]), win=WIN)
        froc["scenario"] = name
        frocs.append(froc)
        tg["scenario"] = name
        tg.to_csv(f"results/ml_target_scores_{name}.csv", index=False)
        for fp in (0.25, 0.5, 1.0):
            rc, rc2, tau = recall_at_fp(froc, fp)
            summ.append(dict(scenario=name, max_false_peaks_per_scene=fp, recall=rc, recall_R_ge_2km=rc2, tau=tau))
        # recall by target radius at the 1-FP/scene operating point
        _, _, tau1 = recall_at_fp(froc, 1.0)
        tg["R_bin"] = pd.cut(tg.R, [1400, 2000, 2800, 4100], labels=["1.5-2 km", "2-2.8 km", "2.8-4 km"])
        by = tg.groupby("R_bin", observed=True).score.apply(lambda s: (s >= tau1).mean())
        for k, v in by.items():
            summ.append(dict(scenario=name, max_false_peaks_per_scene=1.0, recall=v, R_bin=str(k), tau=tau1))
    pd.concat(frocs).to_csv("results/ml_froc.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv("results/ml_object_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(s.round(3).to_string(index=False))


def froc_generic(fe, S, T, win=WIN):
    """FROC for arbitrary score maps (no probability scale): thresholds from peak-score quantiles."""
    from scipy.ndimage import maximum_filter
    from geometry import inside
    ny, nx = fe.shape
    tgt, false = [], []
    for s, tg in zip(S, T):
        grid = np.full(ny * nx, -np.inf)
        grid[fe.inside] = s
        grid = grid.reshape(ny, nx)
        pk = (grid == maximum_filter(grid, size=win, mode="constant", cval=-np.inf)) & np.isfinite(grid)
        pxy, pv = fe.cells_all[pk.ravel()], grid[pk]
        matched = np.zeros(len(pv), bool)
        for x0, y0, R, dr in tg:
            hit = np.hypot(pxy[:, 0] - x0, pxy[:, 1] - y0) <= max(R, 1000.0)
            matched |= hit
            if inside(fe.hull, [[x0, y0]])[0]:
                tgt.append(pv[hit].max() if hit.any() else -np.inf)
        false.extend(pv[~matched].tolist())
    tgt, false = np.array(tgt), np.array(false)
    thr = np.unique(np.quantile(np.concatenate([tgt[np.isfinite(tgt)], false]), np.linspace(0, 1, 400)))
    rec = np.array([(tgt >= t).mean() for t in thr])
    fp = np.array([(false >= t).sum() / len(S) for t in thr])
    return rec, fp


def baselines(name="A_clutter"):
    from features import FEATURES
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    z = np.load(f"results/test_{name}.npz", allow_pickle=True)
    T = list(z["targets"])
    maps = {"ML (HGB ensemble)": z["p"].reshape(-1, fe.n_in)}
    X = np.concatenate([fe.transform(z["D"][:, i:i + 250]) for i in range(0, z["D"].shape[1], 250)])
    maps["map value (TPS)"] = X[..., FEATURES.index("tps")]
    maps["annulus contrast 3-6 km"] = X[..., FEATURES.index("con_3_6")]
    maps["matched filter R = 2.5 km"] = X[..., FEATURES.index("mf_R2.5")]
    maps["-Laplacian (SVD proxy)"] = -X[..., FEATURES.index("lap")]
    rows = []
    for k, S in maps.items():
        rec, fp = froc_generic(fe, S, T)
        for lvl in (0.5, 1.0, 2.0):
            ok = fp <= lvl
            rows.append(dict(scenario=name, method=k, max_false_peaks_per_scene=lvl,
                             recall=rec[ok].max() if ok.any() else 0.0))
        np.savez_compressed(f"results/froc_curve_{name}_{k.split()[0].replace('-', 'neg')}.npz", recall=rec, fp=fp)
    out = pd.DataFrame(rows)
    out.to_csv(f"results/froc_baselines_{name}.csv", index=False)
    print(out.pivot_table(index="method", columns="max_false_peaks_per_scene", values="recall").round(3).to_string())


if __name__ == "__main__":
    main()
    for name in SCENARIOS:
        baselines(name)
