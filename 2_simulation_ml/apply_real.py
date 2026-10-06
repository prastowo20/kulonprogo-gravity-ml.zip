"""Apply the simulation-trained models to the real PSG terrestrial gravity data.

Outputs a probability map of complex-scale eruption-centre zones with four uncertainty layers:
model (ensemble spread), data noise (perturbation), station influence (leave-one-station-out)
and prior (spread between prior scenarios A/B/C). Also checks covariate shift between real and
synthetic features and reports candidate peaks.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

from features import FEATURES, FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import REFERENCE, SCENARIOS

N_MODELS = 5
SIGMA_PERT, N_PERT = 2.0, 200
GAJAH = (401448.0, 9140279.0)          # reference summit point used in earlier work


def predict(models, X):
    B, n, F = X.shape
    return np.stack([m.predict_proba(X.reshape(-1, F))[:, 1].reshape(B, n) for m in models])  # (M, B, n)


def peaks(fe, p, pmin=0.2, win=7):
    ny, nx = fe.shape
    grid = np.full(ny * nx, -np.inf)
    grid[fe.inside] = p
    grid = grid.reshape(ny, nx)
    pk = (grid == maximum_filter(grid, size=win, mode="constant", cval=-np.inf)) & (grid >= pmin)
    idx_all = np.where(pk.ravel())[0]
    inside_idx = np.cumsum(fe.inside) - 1
    return idx_all, inside_idx[idx_all]


def main():
    xy, df = load_psg("data/gravity_requested_area.csv")
    d = df.CBA.to_numpy()
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    X = fe.transform(d)
    ens = [joblib.load(f"models/hgb_{REFERENCE}_m{m}.joblib") for m in range(N_MODELS)]
    pm = predict(ens, X)[:, 0]                              # (M, n)
    pA, pA_sd = pm.mean(0), pm.std(0)
    scen = {REFERENCE: pA}
    for name in SCENARIOS:
        if name != REFERENCE:
            scen[name] = predict([joblib.load(f"models/hgb_{name}_m0.joblib")], X)[0, 0]
    S = np.stack(list(scen.values()))
    prior_range = S.max(0) - S.min(0)

    rng = np.random.default_rng(20260925)
    pert = d[:, None] + SIGMA_PERT * rng.standard_normal((len(d), N_PERT))
    pp = predict(ens, fe.transform(pert)).mean(0)          # (N_PERT, n)
    noise_sd = pp.std(0)

    jk = []
    for i in range(len(d)):
        keep = np.arange(len(d)) != i
        fei = FeatureExtractor(xy[keep], hull)             # same hull -> same cells
        jk.append(predict(ens, fei.transform(d[keep])).mean(0)[0])
    jk = np.array(jk)
    jk_sd = jk.std(0) * np.sqrt(len(d) - 1)                # jackknife standard error
    influence = np.abs(jk - pA).mean(1)

    # covariate shift: share of real cells outside the synthetic 0.5-99.5% range, per feature
    z = np.load(f"results/test_{REFERENCE}.npz", allow_pickle=True)
    Xs = fe.transform(z["D"][:, :300]).reshape(-1, len(FEATURES))
    lo, hi = np.nanpercentile(Xs, 0.5, axis=0), np.nanpercentile(Xs, 99.5, axis=0)
    Xr = X[0]
    shift = pd.DataFrame(dict(feature=FEATURES, real_min=np.nanmin(Xr, 0), real_max=np.nanmax(Xr, 0),
                              syn_p0_5=lo, syn_p99_5=hi,
                              frac_outside=[np.nanmean((Xr[:, j] < lo[j]) | (Xr[:, j] > hi[j]))
                                            for j in range(len(FEATURES))]))
    shift.to_csv("results/real_covariate_shift.csv", index=False)

    ia, ii = peaks(fe, pA)
    pk = pd.DataFrame(dict(x=fe.cells_all[ia, 0], y=fe.cells_all[ia, 1], p=pA[ii], p_ens_sd=pA_sd[ii],
                           p_noise_sd=noise_sd[ii], p_jackknife_se=jk_sd[ii],
                           **{f"p_{k}": v[ii] for k, v in scen.items() if k != REFERENCE}))
    pk["dist_to_gajah_km"] = np.hypot(pk.x - GAJAH[0], pk.y - GAJAH[1]) / 1e3
    ch = jk[:, ii] - pA[ii][None, :]                        # (n_st, n_peaks) change when a station is removed
    k = np.abs(ch).argmax(0)
    pk["p_jk_min"], pk["p_jk_max"] = jk[:, ii].min(0), jk[:, ii].max(0)
    pk["key_station_x"], pk["key_station_y"], pk["key_station_cba"] = xy[k, 0], xy[k, 1], d[k]
    pk["key_station_change"] = ch[k, np.arange(len(ii))]
    pk = pk.sort_values("p", ascending=False)
    pk.to_csv("results/real_peaks.csv", index=False)

    g = np.argmin(np.hypot(fe.cells[:, 0] - GAJAH[0], fe.cells[:, 1] - GAJAH[1]))
    summary = dict(p_at_gajah=float(pA[g]), p_at_gajah_scenarios={k: float(v[g]) for k, v in scen.items()},
                   nearest_peak_to_gajah_km=float(pk.dist_to_gajah_km.min()) if len(pk) else None,
                   area_p_ge_0_5_km2=float((pA >= 0.5).sum() * 0.25),
                   area_p_ge_0_3_km2=float((pA >= 0.3).sum() * 0.25),
                   max_p=float(pA.max()), mean_ens_sd=float(pA_sd.mean()), mean_noise_sd=float(noise_sd.mean()),
                   mean_jackknife_se=float(jk_sd.mean()), mean_prior_range=float(prior_range.mean()),
                   most_influential_stations=[dict(x=float(xy[i, 0]), y=float(xy[i, 1]), cba=float(d[i]),
                                                   mean_abs_change=float(influence[i]))
                                              for i in np.argsort(-influence)[:5]])
    json.dump(summary, open("results/real_summary.json", "w"), indent=1)
    np.savez_compressed("results/real_maps.npz", cells=fe.cells, pA=pA, pA_sd=pA_sd, noise_sd=noise_sd,
                        jk_se=jk_sd, prior_range=prior_range, influence=influence,
                        **{f"p_{k}": v for k, v in scen.items()})
    print(json.dumps(summary, indent=1))
    print(pk.head(10).round(3).to_string(index=False))
    print(shift[shift.frac_outside > 0].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
