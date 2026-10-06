"""Head-to-head comparison of ML methods under one protocol.

Every method sees the same synthetic training scenes (prior A, 4,000 scenes x 300 cells), is
tuned on the same scene-wise validation set (800 scenes) by average precision, and is tested on
the same 1,000 held-out scenes (all cells). Transfer to priors B and C tests robustness to prior
misspecification. The fitted models are then applied to the real PSG data, with a global test
against 1,000 target-free scenes.

Usage: python compare_models.py [model ...]   (default: all)
"""
from __future__ import annotations

import copy
import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import FEATURES, FeatureExtractor
from geometry import convex_hull, inside, load_psg
from priors_cfg import SCENARIOS
from scenes import ProfileBank, SceneGenerator

SEED = 20260924
GAJAH = (401448.0, 9140279.0)
warnings.filterwarnings("ignore", category=UserWarning)

LABELS = {"LR": "Logistic regression", "DT": "Decision tree", "RF": "Random forest", "ET": "Extra trees",
          "HGB": "Gradient boosting (HGB)", "XGB": "XGBoost", "SVM": "SVM (RBF, Nystroem)",
          "MLP": "Neural network (MLP)", "GMM": "Gaussian mixture (unsupervised)"}


def load(name, keys=("X", "y", "targets")):
    z = np.load(f"results/ds_{name}.npz", allow_pickle=True)
    return {k: z[k] for k in keys if k in z.files}


def ece(y, p):
    k = np.clip((p * 10).astype(int), 0, 9)
    return float(sum(abs(y[k == i].mean() - p[k == i].mean()) * (k == i).mean() for i in range(10) if (k == i).any()))


def froc(fe, S, T, win=7):
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
    return {lvl: float(rec[fp <= lvl].max()) if (fp <= lvl).any() else 0.0 for lvl in (0.5, 1.0, 2.0)}


# ------------------------------------------------------------------ model builders
def imp_scale():
    return [SimpleImputer(strategy="median"), StandardScaler()]


def fit_LR(tr, va):
    best = None
    for C in (0.01, 0.1, 1.0):
        m = make_pipeline(*imp_scale(), LogisticRegression(C=C, max_iter=3000))
        m.fit(tr["X"], tr["y"])
        ap = average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1])
        if best is None or ap > best[0]:
            best = (ap, m, dict(C=C))
    return best


def fit_DT(tr, va):
    best = None
    for d in (6, 10, 14):
        m = make_pipeline(SimpleImputer(strategy="median"),
                          DecisionTreeClassifier(max_depth=d, min_samples_leaf=200, random_state=SEED))
        m.fit(tr["X"], tr["y"])
        ap = average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1])
        if best is None or ap > best[0]:
            best = (ap, m, dict(max_depth=d))
    return best


def fit_forest(cls):
    def f(tr, va):
        m = make_pipeline(SimpleImputer(strategy="median"),
                          cls(n_estimators=200, max_features="sqrt", min_samples_leaf=50, max_samples=0.2,
                              bootstrap=True, n_jobs=2, random_state=SEED))
        m.fit(tr["X"], tr["y"])
        return average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1]), m, dict(
            n_estimators=200, min_samples_leaf=50, max_samples=0.2)
    return f


def fit_HGB(tr, va):
    m = joblib.load("models/hgb_A_clutter_m0.joblib")      # trained on exactly this training set (ML v1)
    return average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1]), m, dict(
        n_iter=int(m.n_iter_), learning_rate=0.05, max_leaf_nodes=31, source="ML v1 model m0")


def fit_XGB(tr, va):
    best = None
    for d in (4, 6):
        m = XGBClassifier(tree_method="hist", max_depth=d, learning_rate=0.05, n_estimators=3000, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0, eval_metric="logloss",
                          early_stopping_rounds=50, n_jobs=2, random_state=SEED)
        m.fit(tr["X"], tr["y"], eval_set=[(va["X"], va["y"])], verbose=False)
        ap = average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1])
        if best is None or ap > best[0]:
            best = (ap, m, dict(max_depth=d, n_trees=int(m.best_iteration) + 1))
    return best


def fit_SVM(tr, va):
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr["X"]), 150_000, replace=False)
    best = None
    for g in (0.02, 0.1):
        base = make_pipeline(*imp_scale(), Nystroem(kernel="rbf", gamma=g, n_components=600, random_state=SEED),
                             LinearSVC(C=1.0, dual=False, max_iter=5000))
        base.fit(tr["X"][sub], tr["y"][sub])
        ap = average_precision_score(va["y"], base.decision_function(va["X"]))
        if best is None or ap > best[0]:
            best = (ap, base, dict(gamma=g, C=1.0, n_components=600, train_rows=150_000))
    cal = CalibratedClassifierCV(FrozenEstimator(best[1]), method="sigmoid").fit(va["X"], va["y"])
    return best[0], cal, dict(best[2], calibration="Platt on validation scenes")


def fit_MLP(tr, va):
    pre = make_pipeline(*imp_scale()).fit(tr["X"])
    Xt, Xv = pre.transform(tr["X"]), pre.transform(va["X"])
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=4096, learning_rate_init=1e-3,
                        random_state=SEED)
    rng = np.random.default_rng(SEED)
    best, bad = (np.inf, None, 0), 0
    for ep in range(1, 61):
        o = rng.permutation(len(Xt))
        mlp.partial_fit(Xt[o], tr["y"][o], classes=[0, 1])
        ll = log_loss(va["y"], mlp.predict_proba(Xv)[:, 1])
        if ll < best[0] - 1e-5:
            best, bad = (ll, copy.deepcopy(mlp), ep), 0
        else:
            bad += 1
            if bad >= 6:
                break
    m = make_pipeline(pre, best[1])
    return average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1]), m, dict(
        hidden=(128, 64), alpha=1e-4, epochs=best[2])


class GMMScorer:
    """Unsupervised: posterior of the mixture component with the highest mean of the
    standardised map value + annulus contrast (a 'compact positive anomaly' component)."""

    def __init__(self, n):
        self.n = n

    COLS = [i for i, f in enumerate(FEATURES) if not (f.endswith("_n") or f.endswith("_dist"))]  # data-driven only

    def fit(self, X):
        X = np.asarray(X, np.float64)[:, self.COLS]
        self.pre = make_pipeline(*imp_scale()).fit(X)
        Z = self.pre.transform(X)
        self.gmm = GaussianMixture(self.n, covariance_type="full", reg_covar=1e-3, random_state=SEED,
                                   max_iter=300).fit(Z)
        names = [FEATURES[i] for i in self.COLS]
        j = [names.index("tps"), names.index("con_3_6")]
        self.k = int(np.argmax(self.gmm.means_[:, j].mean(1)))
        self.bic = self.gmm.bic(Z)
        return self

    def predict_proba(self, X):
        p = self.gmm.predict_proba(self.pre.transform(np.asarray(X, np.float64)[:, self.COLS]))[:, self.k]
        return np.column_stack([1 - p, p])


def fit_GMM(tr, va):
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(tr["X"]), 200_000, replace=False)
    fits = [GMMScorer(n).fit(tr["X"][sub]) for n in (3, 5, 8)]
    m = min(fits, key=lambda f: f.bic)                     # label-free model selection
    return average_precision_score(va["y"], m.predict_proba(va["X"])[:, 1]), m, dict(
        n_components=m.n, selection="BIC (no labels)")


BUILDERS = {"LR": fit_LR, "DT": fit_DT, "RF": fit_forest(RandomForestClassifier),
            "ET": fit_forest(ExtraTreesClassifier), "HGB": fit_HGB, "XGB": fit_XGB, "SVM": fit_SVM,
            "MLP": fit_MLP, "GMM": fit_GMM}


def predict(m, X, chunk=100_000):
    return np.concatenate([m.predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])


def main():
    todo = sys.argv[1:] or list(BUILDERS)
    xy, df = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    tr, va, te = load("A_train"), load("A_val"), load("A_test")
    tests = {"A": te, "B": load("B_test"), "C": load("C_test")}
    Xreal = fe.transform(df.CBA.to_numpy())[0]
    # target-free scenes for the global test (same generator as ML v1 global_test.py, prior A)
    gen = SceneGenerator(xy, hull, priors=dict(SCENARIOS["A_clutter"], n_targets_p=[1.0, 0, 0, 0]),
                         bank=ProfileBank("models/profile_bank.npz"))
    rng0 = np.random.default_rng([4242, 0])
    Xnull = fe.transform(np.column_stack([gen.sample(rng0).d for _ in range(1000)])).reshape(-1, len(FEATURES))
    g = np.argmin(np.hypot(fe.cells[:, 0] - GAJAH[0], fe.cells[:, 1] - GAJAH[1]))

    for name in todo:
        t0 = time.time()
        ap_val, m, params = BUILDERS[name](tr, va)
        t_fit = time.time() - t0
        row = dict(model=name, label=LABELS[name], val_AP=ap_val, fit_s=round(t_fit, 1), params=json.dumps(params))
        for key, d in tests.items():
            t1 = time.time()
            p = predict(m, d["X"])
            if key == "A":
                row["predict_s_per_Mrow"] = round((time.time() - t1) / len(p) * 1e6, 2)
                np.save(f"results/pred_{name}_A.npy", p.astype(np.float32))
            y = d["y"]
            row[f"{key}_AUC"] = roc_auc_score(y, p)
            row[f"{key}_AP"] = average_precision_score(y, p)
            if key == "A":
                row["A_brier"] = brier_score_loss(y, p)
                row["A_logloss"] = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
                row["A_ECE"] = ece(y, p)
            fr = froc(fe, p.reshape(-1, fe.n_in), list(d["targets"]))
            for lvl, v in fr.items():
                row[f"{key}_recall@{lvl}FP"] = v
        pr = predict(m, Xreal)
        pn = predict(m, Xnull).reshape(1000, fe.n_in).max(1)
        row.update(real_max_p=float(pr.max()), real_argmax_x=float(fe.cells[pr.argmax(), 0]),
                   real_argmax_y=float(fe.cells[pr.argmax(), 1]), real_p_gajah=float(pr[g]),
                   global_p_value=float((pn >= pr.max()).mean()))
        np.save(f"results/real_{name}.npy", pr.astype(np.float32))
        with open(f"results/cmp_{name}.json", "w") as f:
            json.dump(row, f, indent=1)
        print(f"{name}: val AP {ap_val:.3f} | A AUC {row['A_AUC']:.3f} AP {row['A_AP']:.3f} "
              f"R@1FP {row['A_recall@1.0FP']:.2f} | B AP {row['B_AP']:.3f} C AP {row['C_AP']:.3f} | "
              f"real max p {row['real_max_p']:.2f} global p {row['global_p_value']:.3f} | fit {t_fit:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
