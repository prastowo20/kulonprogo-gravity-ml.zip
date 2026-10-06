"""Value of information of a GGMplus-like regional product for the ML detector.

HGB is trained with PSG features + emulated GGM features at four error levels (0, 2, 5, 10 mGal)
and compared with the PSG-only model on the same test scenes (emulated GGM with the same error
level). A GGM-only model (regional value, gradient, Laplacian; no terrestrial data) shows what
the global model can do on its own, e.g. outside the PSG window.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_models import froc, load
from features import FeatureExtractor
from geometry import convex_hull, load_psg
from ggm import GGM_FEATURES

GGM_ERR = [0.0, 2.0, 5.0, 10.0]
NG = len(GGM_FEATURES)


def hgb(seed=20260924):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1500, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, scoring="loss",
                                          n_iter_no_change=30, random_state=seed)


def main():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    keys = ("X", "y", "targets", "XG")
    tr, va, te = load("A_train", keys), load("A_val", keys), load("A_test", keys)
    T = list(te["targets"])
    rows = []
    base = joblib.load("models/hgb_A_clutter_m0.joblib")
    p = base.predict_proba(te["X"])[:, 1]
    rows.append(dict(model="PSG only", ggm_err_mGal=np.nan, AUC=roc_auc_score(te["y"], p),
                     AP=average_precision_score(te["y"], p), **{f"recall@{k}FP": v for k, v in
                                                                froc(fe, p.reshape(-1, fe.n_in), T).items()}))
    for lev, e in enumerate(GGM_ERR):
        sl = slice(lev * NG, (lev + 1) * NG)
        for label, cols in [("PSG + GGM", None), ("GGM only (regional)", [0, 1, 2])]:
            if cols is None:
                Xtr, Xva, Xte = (np.hstack([d["X"], d["XG"][:, sl]]) for d in (tr, va, te))
            else:
                Xtr, Xva, Xte = (d["XG"][:, sl][:, cols] for d in (tr, va, te))
            m = hgb().fit(Xtr, tr["y"], X_val=Xva, y_val=va["y"])
            p = m.predict_proba(Xte)[:, 1]
            rows.append(dict(model=label, ggm_err_mGal=e, AUC=roc_auc_score(te["y"], p),
                             AP=average_precision_score(te["y"], p), n_iter=m.n_iter_,
                             **{f"recall@{k}FP": v for k, v in froc(fe, p.reshape(-1, fe.n_in), T).items()}))
            if cols is None:
                joblib.dump(m, f"models/hgb_psg_ggm_e{e:g}.joblib")
            print(rows[-1], flush=True)
    # how well does the emulated GGM track station data at each error level? (for calibration)
    gs = []
    for lev, e in enumerate(GGM_ERR):
        # the residual feature res_nn3 is station data minus GGM; use raw correlation instead:
        pass
    out = pd.DataFrame(rows)
    out.to_csv("results/ggm_value.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
