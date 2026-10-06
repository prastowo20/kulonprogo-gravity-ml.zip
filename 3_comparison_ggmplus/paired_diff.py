"""Paired scene-bootstrap comparison of the nine methods (prior A test, 1000 scenes).

All methods are scored on the same resampled scenes, so the difference to the reference model
(HGB, the ML v1 model) has a much narrower interval than the per-model intervals of
bootstrap_ci.py. Writes results/model_comparison_paired.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bootstrap_ci import MODELS, N_IN, per_scene_peaks, recall_at
from compare_models import load
from features import FeatureExtractor
from geometry import convex_hull, load_psg

REF = "HGB"
B_AP, B_REC = 200, 400


def main():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    te = load("A_test", keys=("y", "targets"))
    Y = te["y"].reshape(-1, N_IN)
    T = list(te["targets"])
    n = len(Y)
    rng = np.random.default_rng(2026)
    boots = [rng.integers(0, n, n) for _ in range(max(B_AP, B_REC))]
    ap, rec, full = {}, {}, {}
    for m in MODELS:
        P = np.load(f"results/pred_{m}_A.npy").reshape(-1, N_IN)
        ap[m] = np.array([average_precision_score(Y[b].ravel(), P[b].ravel()) for b in boots[:B_AP]])
        pk = per_scene_peaks(fe, P, T)
        rec[m] = np.array([recall_at(pk, b) for b in boots[:B_REC]])
        full[m] = (average_precision_score(Y.ravel(), P.ravel()), recall_at(pk, np.arange(n)))
        print(m, full[m], flush=True)
    rows = []
    for m in MODELS:
        dap, drec = ap[m] - ap[REF], rec[m] - rec[REF]
        rows.append(dict(model=m, AP=full[m][0], dAP_vs_HGB=full[m][0] - full[REF][0],
                         dAP_lo=np.percentile(dap, 2.5), dAP_hi=np.percentile(dap, 97.5),
                         recall1=full[m][1], dRecall1_vs_HGB=full[m][1] - full[REF][1],
                         dRecall1_lo=np.percentile(drec, 2.5), dRecall1_hi=np.percentile(drec, 97.5),
                         P_better_AP=float((dap > 0).mean()), P_better_recall1=float((drec > 0).mean())))
    out = pd.DataFrame(rows)
    out.to_csv("results/model_comparison_paired.csv", index=False)
    print(out.round(3).to_string(index=False))


def ggm_main():
    """Paired differences of the PSG + GGM models (calibrated emulation) vs PSG only (HGB m0)."""
    import glob
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    te = load("A_test", keys=("y", "targets"))
    Y = te["y"].reshape(-1, N_IN)
    T = list(te["targets"])
    n = len(Y)
    rng = np.random.default_rng(2026)
    boots = [rng.integers(0, n, n) for _ in range(max(B_AP, B_REC))]
    files = {"PSG only (HGB m0)": "results/pred_HGB_A.npy"}
    files.update({f[len("results/pred_psg_ggm_"):-len("_A.npy")]: f for f in sorted(glob.glob("results/pred_psg_ggm_L*_A.npy"))})
    ap, rec, full = {}, {}, {}
    for k, f in files.items():
        P = np.load(f).reshape(-1, N_IN)
        ap[k] = np.array([average_precision_score(Y[b].ravel(), P[b].ravel()) for b in boots[:B_AP]])
        pk = per_scene_peaks(fe, P, T)
        rec[k] = np.array([recall_at(pk, b) for b in boots[:B_REC]])
        full[k] = (average_precision_score(Y.ravel(), P.ravel()), recall_at(pk, np.arange(n)))
        print(k, full[k], flush=True)
    ref = "PSG only (HGB m0)"
    rows = []
    for k in files:
        dap, drec = ap[k] - ap[ref], rec[k] - rec[ref]
        rows.append(dict(model=k, AP=full[k][0], dAP=full[k][0] - full[ref][0], dAP_lo=np.percentile(dap, 2.5),
                         dAP_hi=np.percentile(dap, 97.5), recall1=full[k][1], dRecall1=full[k][1] - full[ref][1],
                         dRecall1_lo=np.percentile(drec, 2.5), dRecall1_hi=np.percentile(drec, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv("results/ggm_value_paired.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    ggm_main() if "--ggm" in sys.argv else main()
