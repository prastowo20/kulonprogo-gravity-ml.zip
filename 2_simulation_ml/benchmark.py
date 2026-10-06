"""ML detector under the exact single-body protocol of the detectability test (PSG-43):
same 500 body locations, same noise realisations, same family-wise false-alarm rate (5%)
and the same location tolerance max(R, 1 km). Directly comparable with D1 and D2."""
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from cylinder import Profile
from features import FeatureExtractor
from geometry import convex_hull, load_psg, pairwise_dist, sample_inner
from priors_cfg import REFERENCE

SEED_POD = 20260923            # seed of run_pod.py (detectability test)
R_LIST = [250, 500, 750, 1000, 1500, 2000, 3000, 4000]
ZT_LIST = [100, 500, 1000]
H = 2000.0
DRHO_LIST = [300.0]
SIGMA_LIST = [2.0, 5.5]
N_BODY, N_NULL, ALPHA, BUFFER = 500, 4000, 0.05, 2000.0
N_MODELS = 1              # ensemble members are near-identical (test AUC 0.772-0.773); m0 used for speed


def ensemble_predict(models, X):
    B, n, F = X.shape
    flat = X.reshape(-1, F)
    return np.mean([m.predict_proba(flat)[:, 1] for m in models], axis=0).reshape(B, n)


def main():
    t0 = time.time()
    xy, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    models = [joblib.load(f"models/hgb_{REFERENCE}_m{m}.joblib") for m in range(N_MODELS)]
    rng = np.random.default_rng(SEED_POD)
    bodies = sample_inner(hull, BUFFER, N_BODY, rng)
    grng = np.random.default_rng([SEED_POD, 0])       # geometry index 0 = PSG-43 in run_pod.py
    Nnull = grng.standard_normal((len(xy), N_NULL))
    Nsig = grng.standard_normal((len(xy), N_BODY))
    Dbs = pairwise_dist(bodies, xy)

    thr = {}
    for sigma in SIGMA_LIST:
        mx = np.concatenate([ensemble_predict(models, fe.transform(sigma * Nnull[:, i:i + 500])).max(1)
                             for i in range(0, N_NULL, 500)])
        thr[sigma] = float(np.quantile(mx, 1 - ALPHA))
    print("ML thresholds (max probability):", thr, flush=True)

    rows = []
    for R in R_LIST:
        tol = max(float(R), 1000.0)
        for zt in ZT_LIST:
            sig = Profile(float(R), float(zt), float(zt) + H)(Dbs).T
            for drho in DRHO_LIST:
                for sigma in SIGMA_LIST:
                    P = ensemble_predict(models, fe.transform(drho * sig + sigma * Nsig))
                    i = P.argmax(1)
                    v = P[np.arange(N_BODY), i]
                    e = np.hypot(*(fe.cells[i] - bodies).T)
                    det = (v > thr[sigma]) & (e <= tol)
                    rows.append(dict(geometry="PSG-43", R_m=R, zt_m=zt, drho=drho, sigma=sigma, pod_ML=det.mean(),
                                     exceed_ML=(v > thr[sigma]).mean(),
                                     locerr_med_ML=float(np.median(e[det])) if det.any() else np.nan))
        print(f"R={R} done [{time.time() - t0:.0f}s]", flush=True)
    ml = pd.DataFrame(rows)
    ref = pd.read_csv("results/pod_results_detectability.csv")
    ref = ref[ref.geometry == "PSG-43"][["geometry", "R_m", "zt_m", "drho", "sigma", "pod_D1", "pod_D2"]]
    out = ml.merge(ref, on=["geometry", "R_m", "zt_m", "drho", "sigma"], how="left")
    out.to_csv("results/benchmark_ML_vs_D1_D2.csv", index=False)
    pd.Series(thr).to_csv("results/benchmark_ML_thresholds.csv")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
