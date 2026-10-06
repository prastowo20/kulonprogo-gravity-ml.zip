"""Null benchmark for the real probability maps (revision, 30 Sep 2026).

Regenerates the distributions that the global test of ML v1 / ensemble_density.py summarised
but did not store: the maximum probability of the map in 1000 target-free scenes per prior
(same generator, same seeds [4242, prior index], same models). The saved medians, 95th
percentiles and global p-values of results/ensemble_density.json are recomputed and checked.

Also recomputes the station jackknife of the reference ensemble (prior A, H1) exactly as in
ML v1 apply_real.py, checks the jackknife standard error against ml_v1/results/real_maps.npz,
and reports the uncertainty layers at P1, P2 and the Gajah summit.

Outputs: results/null_benchmark.npz (null maxima per prior, jackknife maps) and
results/null_benchmark.json (summary + reproduction checks).
"""
from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from features import FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import SCENARIOS
from scenes import ProfileBank, SceneGenerator

N = 1000
GAJAH = (401448.0, 9140279.0)
P1, P2 = (400000.0, 9145000.0), (402000.0, 9136000.0)
V1 = os.environ.get("ML_V1_RESULTS", "../2_simulation_ml/results")   # outputs of stage 2


def predict(models, X):
    B, n, F = X.shape
    return np.mean([m.predict_proba(X.reshape(-1, F))[:, 1].reshape(B, n) for m in models], axis=0)


def main():
    t0 = time.time()
    xy, df = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    dens = pd.read_csv("data/derived/psg_cba_H1_H2.csv")
    D = {"H1": dens.CBA_H1.to_numpy(), "H2": dens.CBA_H2_2670.to_numpy()}
    ref = json.load(open("results/ensemble_density.json"))
    bank = ProfileBank("models/profile_bank.npz")
    cell = lambda q: int(np.argmin(np.hypot(fe.cells[:, 0] - q[0], fe.cells[:, 1] - q[1])))
    out, arrays, checks = {}, {"cells": fe.cells}, {}
    for si, (name, pr) in enumerate(SCENARIOS.items()):
        models = ([joblib.load(f"models/hgb_{name}_m{m}.joblib") for m in range(5)] if name == "A_clutter"
                  else [joblib.load(f"models/hgb_{name}_m0.joblib")])
        gen = SceneGenerator(xy, hull, priors=dict(pr, n_targets_p=[1.0, 0.0, 0.0, 0.0]), bank=bank)
        rng = np.random.default_rng([4242, si])
        D0 = np.column_stack([gen.sample(rng).d for _ in range(N)])
        mx0 = np.concatenate([predict(models, fe.transform(D0[:, i:i + 250])).max(1) for i in range(0, N, 250)])
        arrays[f"null_max_{name}"] = mx0
        for h, d in D.items():
            p = predict(models, fe.transform(d))[0]
            r = ref[f"{name} | {h}"]
            s = dict(real_max_p=float(p.max()), global_p=float((mx0 >= p.max()).mean()),
                     null_median=float(np.median(mx0)), null_p95=float(np.quantile(mx0, 0.95)),
                     null_mean=float(mx0.mean()), null_max=float(mx0.max()))
            out[f"{name} | {h}"] = s
            checks[f"{name} | {h}"] = dict(
                real_max_p=abs(s["real_max_p"] - r["max_p"]) < 1e-12,
                global_p=abs(s["global_p"] - r["global_p"]) < 1e-12,
                null_median=abs(s["null_median"] - r["null_max_p_median"]) < 1e-12,
                null_p95=abs(s["null_p95"] - r["null_max_p_p95"]) < 1e-12)
            print(name, h, s, checks[f"{name} | {h}"], f"[{time.time() - t0:.0f}s]", flush=True)

    # station jackknife of the reference ensemble (prior A, H1), as in ML v1 apply_real.py
    ens = [joblib.load(f"models/hgb_A_clutter_m{m}.joblib") for m in range(5)]
    d = D["H1"]
    pA = predict(ens, fe.transform(d))[0]
    jk = []
    for i in range(len(d)):
        keep = np.arange(len(d)) != i
        fei = FeatureExtractor(xy[keep], hull)
        jk.append(predict(ens, fei.transform(d[keep]))[0])
    jk = np.array(jk)
    jk_se = jk.std(0) * np.sqrt(len(d) - 1)
    v1 = np.load(f"{V1}/real_maps.npz")
    checks["jackknife_se_matches_ml_v1"] = bool(np.allclose(jk_se, v1["jk_se"], atol=1e-9))
    checks["reference_map_matches_ml_v1"] = bool(np.allclose(pA, v1["pA"], atol=1e-12))
    arrays["jk_maps_A_H1"] = jk
    locs = {}
    for lab, q in [("P1", P1), ("P2", P2), ("Gajah", GAJAH)]:
        c = cell(q)
        dist = np.hypot(xy[:, 0] - q[0], xy[:, 1] - q[1])
        i_near = int(np.argmin(dist))
        i_key = int(np.argmin(jk[:, c]))
        locs[lab] = dict(cell=[float(v) for v in fe.cells[c]], p=float(pA[c]),
                         ens_sd=float(v1["pA_sd"][c]), noise_sd=float(v1["noise_sd"][c]),
                         jackknife_se=float(jk_se[c]), jackknife_min=float(jk[:, c].min()),
                         jackknife_max=float(jk[:, c].max()), prior_range=float(v1["prior_range"][c]),
                         p_B=float(v1["p_B_mixed"][c]), p_C=float(v1["p_C_body"][c]),
                         nearest_station=[float(v) for v in xy[i_near]], nearest_station_km=float(dist[i_near] / 1e3),
                         nearest_station_cba=float(d[i_near]),
                         p_without_nearest_station=float(jk[i_near, c]),
                         key_station=[float(v) for v in xy[i_key]], key_station_cba=float(d[i_key]),
                         change_without_key_station=float(jk[i_key, c] - pA[c]))
        print(lab, locs[lab], flush=True)
    np.savez_compressed("results/null_benchmark.npz", **arrays)
    json.dump(dict(null=out, locations=locs, reproduction_checks=checks, n_null=N,
                   seeds="null scenes: numpy default_rng([4242, prior index]), as ML v1 global_test.py"),
              open("results/null_benchmark.json", "w"), indent=1)

    print("checks:", checks, f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
