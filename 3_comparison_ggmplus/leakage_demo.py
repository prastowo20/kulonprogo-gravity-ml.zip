"""How much does a random cell-wise train/test split inflate performance?

(1) Single-map protocol (as in Abdulfarraj et al. 2025): one map, cells labelled 1 (centre
    zone) or 0, balanced by subsampling 0s, random stratified 80/10/10 split, random forest.
    Metrics on the 10% test cells of the SAME map versus on 500 NEW maps.
(2) Pooled random split across 150 maps (all cells) versus a scene-wise split.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

SEED = 7
N_IN = 1384


def rf(leaf=1, n=200):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, n_jobs=2, random_state=SEED))


def metrics(y, p, prefix):
    yhat = (p >= 0.5).astype(int)
    return {f"{prefix}_accuracy": accuracy_score(y, yhat), f"{prefix}_balanced_acc": balanced_accuracy_score(y, yhat),
            f"{prefix}_F1": f1_score(y, yhat, zero_division=0), f"{prefix}_AUC": roc_auc_score(y, p),
            f"{prefix}_AP": average_precision_score(y, p)}


def main():
    z = np.load("results/ds_A_test.npz", allow_pickle=True)
    X = z["X"].reshape(-1, N_IN, z["X"].shape[1])
    Y = z["y"].reshape(-1, N_IN)
    new = slice(500, 1000)
    Xn, Yn = X[new].reshape(-1, X.shape[2]), Y[new].ravel()
    rng = np.random.default_rng(SEED)

    # (1) single-map protocol, repeated on 30 maps that contain a sizeable centre zone
    cand = [i for i in range(500) if 40 <= Y[i].sum() <= 400][:30]
    rows = []
    for i in cand:
        pos = np.where(Y[i])[0]
        neg = rng.choice(np.where(~Y[i])[0], len(pos), replace=False)
        idx = np.concatenate([pos, neg])
        Xi, yi = X[i, idx], Y[i, idx].astype(int)
        Xtr, Xtmp, ytr, ytmp = train_test_split(Xi, yi, test_size=0.2, stratify=yi, random_state=SEED)
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED)
        m = rf(leaf=1, n=100).fit(Xtr, ytr)
        r = dict(scene=i, n_pos=len(pos))
        r.update(metrics(yte, m.predict_proba(Xte)[:, 1], "same_map_test"))
        r.update(metrics(Yn, m.predict_proba(Xn)[:, 1], "new_maps"))
        rows.append(r)
    single = pd.DataFrame(rows)
    single.to_csv("results/leakage_single_map.csv", index=False)

    # (2) pooled random split vs scene-wise split (150 maps, all cells)
    Xp, Yp = X[:150].reshape(-1, X.shape[2]), Y[:150].ravel()
    sid = np.repeat(np.arange(150), N_IN)
    tr, te = train_test_split(np.arange(len(Yp)), test_size=0.2, random_state=SEED)
    m_rand = rf(leaf=1).fit(Xp[tr], Yp[tr])
    pooled = dict(protocol="random cell split (leaky)")
    pooled.update(metrics(Yp[te], m_rand.predict_proba(Xp[te])[:, 1], "held_out_cells"))
    pooled.update(metrics(Yn, m_rand.predict_proba(Xn)[:, 1], "new_maps"))
    trs, tes = sid < 120, sid >= 120
    m_scene = rf(leaf=1).fit(Xp[trs], Yp[trs])
    scene = dict(protocol="scene-wise split (honest)")
    scene.update(metrics(Yp[tes], m_scene.predict_proba(Xp[tes])[:, 1], "held_out_cells"))
    scene.update(metrics(Yn, m_scene.predict_proba(Xn)[:, 1], "new_maps"))
    pooled_df = pd.DataFrame([pooled, scene])
    pooled_df.to_csv("results/leakage_pooled.csv", index=False)

    summ = {"single_map_median": single.drop(columns=["scene"]).median().round(3).to_dict(),
            "pooled": pooled_df.set_index("protocol").round(3).to_dict(orient="index")}
    json.dump(summ, open("results/leakage_summary.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(single.drop(columns=["scene"]).median().round(3).to_string())
    print(pooled_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
