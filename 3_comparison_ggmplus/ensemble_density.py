"""Reference model of the paper (HGB ensemble of ML v1, prior A; single HGB for priors B and C)
applied to the terrestrial data under both reduction-density hypotheses (H1: as supplied;
H2: 2840 -> 2670 kg/m^3, psg_density.py), with the global test of ML v1 (1000 target-free
scenes per prior, seeds [4242, prior index]). Reproduces results/global_test.json of ML v1
for H1 as a check.
"""
from __future__ import annotations

import json

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


def predict(models, X):
    B, n, F = X.shape
    return np.mean([m.predict_proba(X.reshape(-1, F))[:, 1].reshape(B, n) for m in models], axis=0)


def main():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    dens = pd.read_csv("data/derived/psg_cba_H1_H2.csv")
    D = {"H1": dens.CBA_H1.to_numpy(), "H2": dens.CBA_H2_2670.to_numpy()}
    bank = ProfileBank("models/profile_bank.npz")
    at = lambda p, q: float(p[np.argmin(np.hypot(fe.cells[:, 0] - q[0], fe.cells[:, 1] - q[1]))])
    out, maps = {}, {}
    for si, (name, pr) in enumerate(SCENARIOS.items()):
        models = ([joblib.load(f"models/hgb_{name}_m{m}.joblib") for m in range(5)] if name == "A_clutter"
                  else [joblib.load(f"models/hgb_{name}_m0.joblib")])
        gen = SceneGenerator(xy, hull, priors=dict(pr, n_targets_p=[1.0, 0.0, 0.0, 0.0]), bank=bank)
        rng = np.random.default_rng([4242, si])
        D0 = np.column_stack([gen.sample(rng).d for _ in range(N)])
        mx0 = np.concatenate([predict(models, fe.transform(D0[:, i:i + 250])).max(1) for i in range(0, N, 250)])
        for h, d in D.items():
            p = predict(models, fe.transform(d))[0]
            maps[f"{name}_{h}"] = p
            out[f"{name} | {h}"] = dict(n_models=len(models), max_p=float(p.max()),
                                        argmax=[float(v) for v in fe.cells[p.argmax()]],
                                        p_P1=at(p, P1), p_P2=at(p, P2), p_gajah=at(p, GAJAH),
                                        area_p_ge_0p3_km2=float((p >= 0.3).sum() * 0.25),
                                        global_p=float((mx0 >= p.max()).mean()),
                                        null_max_p_median=float(np.median(mx0)),
                                        null_max_p_p95=float(np.quantile(mx0, 0.95)))
            print(name, h, out[f"{name} | {h}"], flush=True)
    for name in SCENARIOS:
        out[f"{name} | corr H1-H2 maps"] = float(np.corrcoef(maps[f"{name}_H1"], maps[f"{name}_H2"])[0, 1])
    np.savez_compressed("results/ensemble_density_maps.npz", cells=fe.cells, **maps)
    json.dump(out, open("results/ensemble_density.json", "w"), indent=1)


if __name__ == "__main__":
    main()
