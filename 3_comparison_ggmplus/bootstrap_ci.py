"""Scene-bootstrap 95% confidence intervals for AP and recall at <= 1 false peak/scene (prior A test),
so that differences between methods can be judged against sampling uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.metrics import average_precision_score

from compare_models import load
from features import FeatureExtractor
from geometry import convex_hull, inside, load_psg

MODELS = ["HGB", "XGB", "RF", "ET", "MLP", "SVM", "LR", "DT", "GMM"]
B = 200
N_IN = 1384


def per_scene_peaks(fe, S, T, win=7):
    ny, nx = fe.shape
    out = []
    for s, tg in zip(S, T):
        grid = np.full(ny * nx, -np.inf)
        grid[fe.inside] = s
        grid = grid.reshape(ny, nx)
        pk = (grid == maximum_filter(grid, size=win, mode="constant", cval=-np.inf)) & np.isfinite(grid)
        pxy, pv = fe.cells_all[pk.ravel()], grid[pk]
        matched = np.zeros(len(pv), bool)
        tscore = []
        for x0, y0, R, dr in tg:
            hit = np.hypot(pxy[:, 0] - x0, pxy[:, 1] - y0) <= max(R, 1000.0)
            matched |= hit
            if inside(fe.hull, [[x0, y0]])[0]:
                tscore.append(pv[hit].max() if hit.any() else -np.inf)
        out.append((np.array(tscore), pv[~matched]))
    return out


def recall_at(peaks, idx, fp_level=1.0):
    tg = np.concatenate([peaks[i][0] for i in idx])
    fa = np.concatenate([peaks[i][1] for i in idx])
    thr = np.unique(np.quantile(np.concatenate([tg[np.isfinite(tg)], fa]), np.linspace(0, 1, 300)))
    rec = np.array([(tg >= t).mean() for t in thr])
    fp = np.array([(fa >= t).sum() / len(idx) for t in thr])
    ok = fp <= fp_level
    return rec[ok].max() if ok.any() else 0.0


def main():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    te = load("A_test", keys=("y", "targets"))
    Y = te["y"].reshape(-1, N_IN)
    T = list(te["targets"])
    n = len(Y)
    rng = np.random.default_rng(99)
    boots = [rng.integers(0, n, n) for _ in range(B)]
    rows = []
    for m in MODELS:
        P = np.load(f"results/pred_{m}_A.npy").reshape(-1, N_IN)
        aps = [average_precision_score(Y[b].ravel(), P[b].ravel()) for b in boots[:100]]
        peaks = per_scene_peaks(fe, P, T)
        recs = [recall_at(peaks, b) for b in boots]
        rows.append(dict(model=m, AP=average_precision_score(Y.ravel(), P.ravel()), AP_lo=np.percentile(aps, 2.5),
                         AP_hi=np.percentile(aps, 97.5), recall1=recall_at(peaks, np.arange(n)),
                         recall1_lo=np.percentile(recs, 2.5), recall1_hi=np.percentile(recs, 97.5)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("results/model_comparison_ci.csv", index=False)


if __name__ == "__main__":
    main()
