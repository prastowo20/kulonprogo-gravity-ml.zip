"""Train and evaluate simulation-trained classifiers of eruption-centre zones.

For each prior scenario: independent synthetic scene sets for training, early-stopping
validation and testing (scene-wise split: no scene contributes to more than one set).
Reference scenario: 5-member HGB ensemble (independent training sets) + LR and RF baselines
+ single-feature physics baselines. Other scenarios: one HGB model each.

Usage: python train.py [--quick]
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, FeatureExtractor
from geometry import convex_hull, inside, load_psg
from priors_cfg import REFERENCE, SCENARIOS
from scenes import ProfileBank, SceneGenerator, cell_labels

QUICK = "--quick" in sys.argv
SEED = 20260924
N_TRAIN, N_VAL, N_TEST, CELLS = (300, 100, 100, 150) if QUICK else (4000, 800, 1000, 300)
N_ENSEMBLE = 2 if QUICK else 5
BATCH = 250


def scene_block(gen, fe, n, rng, cells=None):
    """Generate n scenes; return features/labels (all cells or `cells` random cells per scene)."""
    Xs, ys, sid, D, T = [], [], [], [], []
    for b0 in range(0, n, BATCH):
        sc = [gen.sample(rng) for _ in range(min(BATCH, n - b0))]
        Db = np.column_stack([s.d for s in sc])
        X = fe.transform(Db)
        for j, s in enumerate(sc):
            y = cell_labels(fe.cells, s.targets)
            idx = np.arange(fe.n_in) if cells is None else rng.choice(fe.n_in, cells, replace=False)
            Xs.append(X[j, idx])
            ys.append(y[idx])
            sid.append(np.full(len(idx), b0 + j))
            T.append(s.targets)
        D.append(Db)
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(sid), np.hstack(D), T)


def fit_hgb(Xtr, ytr, Xva, yva, seed):
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1500, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                         scoring="loss", n_iter_no_change=30, random_state=seed)
    clf.fit(Xtr, ytr, X_val=Xva, y_val=yva)
    return clf


def cell_metrics(y, p):
    bins = np.linspace(0, 1, 11)
    k = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = sum(abs(y[k == i].mean() - p[k == i].mean()) * (k == i).mean() for i in range(10) if (k == i).any())
    return dict(roc_auc=roc_auc_score(y, p), avg_precision=average_precision_score(y, p),
                brier=brier_score_loss(y, p), log_loss=log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), ece=ece,
                prevalence=y.mean())


def reliability(y, p, nb=10):
    bins = np.linspace(0, 1, nb + 1)
    k = np.clip(np.digitize(p, bins) - 1, 0, nb - 1)
    return pd.DataFrame([dict(bin_lo=bins[i], bin_hi=bins[i + 1], n=int((k == i).sum()),
                              p_mean=p[k == i].mean() if (k == i).any() else np.nan,
                              obs_freq=y[k == i].mean() if (k == i).any() else np.nan) for i in range(nb)])


def object_eval(fe, P, T, win=3, pmin=0.02):
    """Free-response evaluation. Peaks = local maxima (win x win cells) of the probability map.
    Each target whose axis lies inside the hull gets a score = highest peak within max(R, 1 km)
    (0 if none). A peak is false if it lies farther than max(R, 1 km) from every target."""
    ny, nx = fe.shape
    ins = fe.inside
    tgt_rows, false_scores = [], []
    for p, tg in zip(P, T):
        grid = np.full(ny * nx, -np.inf)
        grid[ins] = p
        grid = grid.reshape(ny, nx)
        pk = (grid == maximum_filter(grid, size=win, mode="constant", cval=-np.inf)) & (grid >= pmin)
        pxy = fe.cells_all[pk.ravel()]
        pv = grid[pk]
        matched = np.zeros(len(pxy), bool)
        for x0, y0, R, dr in tg:
            dd = np.hypot(pxy[:, 0] - x0, pxy[:, 1] - y0)
            hit = dd <= max(R, 1000.0)
            matched |= hit
            if inside(fe.hull, [[x0, y0]])[0]:
                tgt_rows.append(dict(R=R, drho=dr, score=pv[hit].max() if hit.any() else 0.0))
        false_scores.extend(pv[~matched].tolist())
    tg_df = pd.DataFrame(tgt_rows)
    fs = np.array(false_scores)
    froc = pd.DataFrame([dict(tau=tau, recall=(tg_df.score >= tau).mean(),
                              recall_R_ge_2km=(tg_df[tg_df.R >= 2000].score >= tau).mean(),
                              false_peaks_per_scene=(fs >= tau).sum() / len(P))
                         for tau in np.round(np.arange(0.05, 0.96, 0.05), 2)])
    return tg_df, froc


def recall_at_fp(froc, fp):
    """Recall at the smallest threshold whose false-peak rate does not exceed fp per scene."""
    ok = froc[froc.false_peaks_per_scene <= fp]
    return (ok.iloc[0].recall, ok.iloc[0].recall_R_ge_2km, ok.iloc[0].tau) if len(ok) else (np.nan, np.nan, np.nan)


def main():
    t0 = time.time()
    xy, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    bank = ProfileBank("models/profile_bank.npz")
    tag = "_quick" if QUICK else ""
    summary, rel_tables, obj_tables = [], [], []
    for si, (name, pr) in enumerate(SCENARIOS.items()):
        gen = SceneGenerator(xy, hull, priors=pr, bank=bank)
        n_models = N_ENSEMBLE if name == REFERENCE else 1
        rng_val = np.random.default_rng([SEED, si, 900])
        Xva, yva, *_ = scene_block(gen, fe, N_VAL, rng_val, cells=CELLS)
        rng_te = np.random.default_rng([SEED, si, 901])
        Xte, yte, sid_te, Dte, Tte = scene_block(gen, fe, N_TEST, rng_te, cells=None)
        preds = []
        for m in range(n_models):
            rng_tr = np.random.default_rng([SEED, si, m])
            Xtr, ytr, *_ = scene_block(gen, fe, N_TRAIN, rng_tr, cells=CELLS)
            clf = fit_hgb(Xtr, ytr, Xva, yva, seed=SEED + m)
            joblib.dump(clf, f"models/hgb_{name}_m{m}{tag}.joblib")
            p = clf.predict_proba(Xte)[:, 1]
            preds.append(p)
            summary.append(dict(scenario=name, model=f"HGB_m{m}", n_iter=clf.n_iter_, **cell_metrics(yte, p)))
            print(f"{name} HGB m{m}: iters={clf.n_iter_} AUC={summary[-1]['roc_auc']:.3f} "
                  f"AP={summary[-1]['avg_precision']:.3f}  [{time.time() - t0:.0f}s]", flush=True)
            if name == REFERENCE and m == 0:
                sub = np.random.default_rng(5).choice(len(Xtr), min(len(Xtr), 400_000), replace=False)
                lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=2000))
                lr.fit(Xtr[sub], ytr[sub])
                rf = make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_features="sqrt",
                                                          n_jobs=2, random_state=SEED))
                rf.fit(Xtr[sub], ytr[sub])
                joblib.dump(lr, f"models/lr_{name}{tag}.joblib")
                joblib.dump(rf, f"models/rf_{name}{tag}.joblib")
                for lab, mdl in [("LogReg", lr), ("RandomForest", rf)]:
                    summary.append(dict(scenario=name, model=lab, n_iter=np.nan,
                                        **cell_metrics(yte, mdl.predict_proba(Xte)[:, 1])))
                for f in ["tps", "mf_R2.5", "con_3_6"]:          # single-feature physics baselines
                    s = Xte[:, FEATURES.index(f)]
                    summary.append(dict(scenario=name, model=f"single:{f}", n_iter=np.nan,
                                        roc_auc=roc_auc_score(yte, s), avg_precision=average_precision_score(yte, s),
                                        brier=np.nan, log_loss=np.nan, ece=np.nan, prevalence=yte.mean()))
        pe = np.mean(preds, axis=0)
        if n_models > 1:
            summary.append(dict(scenario=name, model="HGB_ensemble", n_iter=np.nan, **cell_metrics(yte, pe)))
        r = reliability(yte, pe)
        r["scenario"] = name
        rel_tables.append(r)
        tg_df, froc = object_eval(fe, pe.reshape(N_TEST, fe.n_in), Tte)
        froc["scenario"] = name
        tg_df["scenario"] = name
        obj_tables.append(froc)
        tg_df.to_csv(f"results/ml_target_scores_{name}{tag}.csv", index=False)
        for fp in (0.25, 0.5, 1.0):
            rc, rc2, tau = recall_at_fp(froc, fp)
            summary.append(dict(scenario=name, model=f"object_level@{fp}FP/scene", n_iter=np.nan,
                                recall_all=rc, recall_R_ge_2km=rc2, tau=tau))
        rc, rc2, tau = recall_at_fp(froc, 0.5)
        print(f"{name}: recall at 0.5 false peaks/scene = {rc:.2f} (R>=2 km {rc2:.2f}, tau={tau})", flush=True)
        # test-set cross-scenario transfer: reference model on other scenarios' test data
        np.savez_compressed(f"results/test_{name}{tag}.npz", D=Dte, p=pe.astype(np.float32),
                            y=yte, targets=np.array(Tte, dtype=object))
        bank.save()
    pd.DataFrame(summary).to_csv(f"results/ml_metrics{tag}.csv", index=False)
    pd.concat(rel_tables).to_csv(f"results/ml_reliability{tag}.csv", index=False)
    pd.concat(obj_tables).to_csv(f"results/ml_froc{tag}.csv", index=False)
    meta = dict(seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, cells_per_scene=CELLS,
                n_ensemble=N_ENSEMBLE, features=FEATURES, scenarios=SCENARIOS, reference=REFERENCE,
                runtime_s=round(time.time() - t0, 1))
    json.dump(meta, open(f"results/ml_meta{tag}.json", "w"), indent=1, default=str)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
