"""Grouped permutation importance on held-out synthetic test scenes (reference model m0).

Features are permuted jointly within physically meaningful groups; the drop in average
precision (AP) and ROC-AUC is reported (mean and SD over 5 repeats)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import FEATURES, FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import REFERENCE
from scenes import cell_labels

GROUPS = {
    "interpolated map (value)": ["tps"],
    "map derivatives (THDR, Laplacian, VDR, TDR, AS)": ["thdr", "lap", "vdr", "tdr", "asig"],
    "annulus contrast (3-6, 6-10 km)": ["con_3_6", "con_6_10"],
    "station ring means": [f"ring{i}_mean" for i in range(5)] + ["ring_in_minus_out"],
    "nearest-station values": ["nn1_val", "nn3_val"],
    "station geometry (counts, distances)": [f"ring{i}_n" for i in range(5)] + ["nn1_dist", "nn3_dist"],
    "matched-filter bank": ["mf_R1.5", "mf_R2.5", "mf_R3.5"],
}


def main(n_scenes=400, repeats=5):
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    z = np.load(f"results/test_{REFERENCE}.npz", allow_pickle=True)
    D, T = z["D"][:, :n_scenes], z["targets"][:n_scenes]
    X = fe.transform(D).reshape(-1, len(FEATURES))
    y = np.concatenate([cell_labels(fe.cells, t) for t in T])
    clf = joblib.load(f"models/hgb_{REFERENCE}_m0.joblib")
    base_ap = average_precision_score(y, clf.predict_proba(X)[:, 1])
    base_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    rng = np.random.default_rng(11)
    rows = []
    for g, cols in GROUPS.items():
        idx = [FEATURES.index(c) for c in cols]
        aps, aucs = [], []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, idx] = X[perm][:, idx]
            p = clf.predict_proba(Xp)[:, 1]
            aps.append(base_ap - average_precision_score(y, p))
            aucs.append(base_auc - roc_auc_score(y, p))
        rows.append(dict(group=g, n_features=len(cols), drop_AP_mean=np.mean(aps), drop_AP_sd=np.std(aps),
                         drop_AUC_mean=np.mean(aucs), drop_AUC_sd=np.std(aucs)))
    out = pd.DataFrame(rows).sort_values("drop_AP_mean", ascending=False)
    out["base_AP"], out["base_AUC"] = base_ap, base_auc
    out.to_csv("results/importance_grouped.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
