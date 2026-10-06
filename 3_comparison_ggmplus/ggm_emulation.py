"""Realistic GGMplus emulation, calibrated against the real PSG-GGMplus misfit (RQ1 -> RQ2 link).

Why: in ggm_value.py the emulated product was LP_10km(true field) + error. A Gaussian with a 10 km cut-off (amplitude 0.5)
still passes ~24% of the amplitude at 7 km and ~50% at 10 km wavelength, i.e. much of
the signal of the 1.5-4 km radius targets. The "GGM only" model then beat the PSG-only model
(AUC 0.82 vs 0.77) although it never sees a terrestrial station, which proves that the emulation
hands the classifier target-scale information that the real GGMplus does not contain: in
GGMplus everything shorter than the EGM2008 limit (degree 2190, ~18 km full wavelength) is
topography-implied (RTM) and is removed by a consistent complete Bouguer reduction.

Here the emulated product is
    G = a * LP_lambda(true field) + e(s_err, L_err = 12 km),
with lambda in {10 km (previous, optimistic), 18.3 km (EGM2008 limit)}, attenuation a in
{1, 0.5} and s_err in {0, 2, 5, 10, 15} mGal. Each variant is checked against the real data by
computing, for every synthetic test scene, exactly the statistics of the real comparison
(ggm_real.py / psg_density.py) and locating the real value in their distribution (a prior
predictive check for the GGM emulation). The scenes are regenerated with the seeds of
datasets.py (station data verified identical), and the error fields reuse the seeds of
datasets.py so that variants differ only in lambda, a and s_err.

Usage:
    python ggm_emulation.py regen            # node fields for A_train / A_val / A_test (~5 min)
    python ggm_emulation.py calib            # results/ggm_emulation_calibration.csv
    python ggm_emulation.py train L a e ...  # e.g. train 18300 1 0 2 5 10 -> results/ggm_value_realistic.csv
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_models import froc, load
from datasets import BATCH, CELLS, N_TEST, N_TRAIN, N_VAL, SEED
from features import FeatureExtractor
from geometry import convex_hull, load_psg
from ggm import GGMGrid
from ggm_value import hgb
from priors_cfg import SCENARIOS
from scenes import ProfileBank, SceneGenerator

ERR_LEVELS = [0.0, 2.0, 5.0, 10.0, 15.0]          # index = 'lev' in the error seed (as datasets.py)
KEYS = {"A_train": ([SEED, 0, 0], N_TRAIN, CELLS), "A_val": ([SEED, 0, 900], N_VAL, CELLS),
        "A_test": ([SEED, 0, 901], N_TEST, None)}
LAMBDA_EGM = 18_300.0


def setup():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    return xy, hull, fe, GGMGrid(fe)


def regen():
    t0 = time.time()
    xy, hull, fe, gg = setup()
    bank = ProfileBank("models/profile_bank.npz")
    gen = SceneGenerator(xy, hull, priors=SCENARIOS["A_clutter"], bank=bank)
    for name, (key, n, cells) in KEYS.items():
        rng = np.random.default_rng(key)
        Ds, Fs, IDX = [], [], []
        for b0 in range(0, n, BATCH):
            sc = [gen.sample(rng, extra_xy=gg.nodes) for _ in range(min(BATCH, n - b0))]
            for _ in sc:                                   # same random stream as datasets.build
                IDX.append(np.arange(fe.n_in) if cells is None else rng.choice(fe.n_in, cells, replace=False))
            Ds.append(np.column_stack([s.d for s in sc]))
            Fs.append(np.column_stack([s.field_extra for s in sc]).astype(np.float32))
        D, F = np.hstack(Ds), np.hstack(Fs)
        Dsaved = load(name, keys=("D",))["D"]
        err = float(np.abs(D - Dsaved).max())
        assert err < 1e-8, f"{name}: regenerated station data differ from datasets.py ({err})"
        np.savez(f"results/F_{name}.npz", F=F, D=D, idx=np.array(IDX, dtype=np.int32))
        print(f"{name}: {n} scenes regenerated, max |D - saved D| = {err:.1e}, {time.time() - t0:.0f}s", flush=True)


def emulate(gg, F, key, i0, lam, a, s_err):
    """F: (n_nodes, B) noise-free fields of scenes i0 .. i0+B-1 -> emulated GGM grids (n_nodes, B)."""
    G = a * gg.lowpass(F.astype(float), lam)
    if s_err > 0:
        lev = ERR_LEVELS.index(s_err)
        G += np.column_stack([gg.error_field(np.random.default_rng([key[0], key[1], 77, lev, i0 + j]), s_err)
                              for j in range(F.shape[1])])
    return G


def _corr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def misfit_stats(gg, xy, D, G):
    """The statistics of the real PSG-GGMplus comparison, per scene (columns)."""
    return misfit_stats_st(xy, D, gg.Ws @ G, gg.Ws @ gg.lowpass(G, 10_000.0), gg.Ws @ gg.lowpass(G, LAMBDA_EGM))


def misfit_stats_st(xy, D, Gs, Gs10, Gs18):
    """As misfit_stats, from the emulated product at the stations: unfiltered (Gs) and after the
    10 km (Gs10) and 18.3 km (Gs18) grid low-pass."""
    c = xy.mean(0)
    u, w = (xy[:, 0] - c[0]) / 1e4, (xy[:, 1] - c[1]) / 1e4
    A = np.column_stack([np.ones(len(xy)), u, w, u * u, u * w, w * w])
    Pr = A @ np.linalg.pinv(A)
    tp, tg = Pr @ D, Pr @ Gs
    out = dict(r_all=_corr(D, Gs), trend_r=_corr(tp, tg), trend_sd_ratio=tg.std(0) / tp.std(0),
               resid_r=_corr(D - tp, Gs - tg), resid_sd_ratio=(Gs - tg).std(0) / (D - tp).std(0))
    Dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    for lam, tag, v in [(10_000.0, "lp10", Gs10), (LAMBDA_EGM, "lp18", Gs18)]:
        sig = lam * np.sqrt(2 * np.log(2)) / (2 * np.pi)
        W = np.exp(-0.5 * (Dd / sig) ** 2)
        W /= W.sum(1, keepdims=True)
        p = W @ D
        out[f"{tag}_r"] = _corr(p, v)
        out[f"{tag}_sd_diff"] = (p - v).std(0)
        out[f"{tag}_sd_ratio"] = v.std(0) / p.std(0)
    return out


def real_stats(hyp="H1_psg_2670"):
    r = json.load(open("results/psg_density.json"))[hyp]
    t, s = r["2nd-order trend"], r["residual after trend"]
    return dict(r_all=r["GGM CBA (all wavelengths)"]["r"], trend_r=t["r"], trend_sd_ratio=t["ggm_trend_sd"] / t["psg_trend_sd"],
                resid_r=s["r"], resid_sd_ratio=s["ggm_sd"] / s["psg_sd"],
                lp10_r=r["band_lowpass_10km"]["r"], lp10_sd_diff=r["band_lowpass_10km"]["sd_diff"],
                lp10_sd_ratio=r["band_lowpass_10km"]["ggm_sd"] / r["band_lowpass_10km"]["psg_sd"],
                lp18_r=r["band_lowpass_EGM2008_limit_18.3km"]["r"],
                lp18_sd_diff=r["band_lowpass_EGM2008_limit_18.3km"]["sd_diff"],
                lp18_sd_ratio=r["band_lowpass_EGM2008_limit_18.3km"]["ggm_sd"] / r["band_lowpass_EGM2008_limit_18.3km"]["psg_sd"])


def calib():
    xy, hull, fe, gg = setup()
    z = np.load("results/F_A_test.npz")
    F, D = z["F"], z["D"]
    key = KEYS["A_test"][0]
    real = {h: real_stats(h) for h in ("H1_psg_2670", "H2_psg_2840_converted")}
    rows = []
    for lam in (10_000.0, LAMBDA_EGM):
        for a in (1.0, 0.5):
            for e in ERR_LEVELS:
                G = emulate(gg, F, key, 0, lam, a, e)
                st = misfit_stats(gg, xy, D, G)
                row = dict(lambda_km=lam / 1e3, a=a, s_err=e)
                worst = 50.0
                for k, v in st.items():
                    row[f"{k}_med"] = float(np.nanmedian(v))
                    row[f"{k}_p05"] = float(np.nanpercentile(v, 5))
                    row[f"{k}_p95"] = float(np.nanpercentile(v, 95))
                    pr = float((v < real["H1_psg_2670"][k]).mean() * 100)
                    row[f"{k}_realpct"] = pr
                    worst = min(worst, pr, 100 - pr)
                row["min_tail_pct_H1"] = worst                # smallest tail probability over all statistics
                w2 = 50.0
                for k, v in st.items():
                    pr = float((v < real["H2_psg_2840_converted"][k]).mean() * 100)
                    w2 = min(w2, pr, 100 - pr)
                row["min_tail_pct_H2"] = w2
                rows.append(row)
                print(f"lam {lam/1e3:4.1f} a {a:.1f} e {e:4.1f}: min tail {worst:5.1f}%  " +
                      " ".join(f"{k}={row[k+'_med']:.2f}({row[k+'_realpct']:.0f})" for k in st), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/ggm_emulation_calibration.csv", index=False)
    json.dump(real, open("results/ggm_real_stats_for_calibration.json", "w"), indent=1)


# ---------------------------------------------------------------- calibration, stage 2
LAMS2 = [10_000.0, LAMBDA_EGM, 30_000.0, 45_000.0]
A2 = [1.0, 0.5, 0.33]
SL2 = [0.0, 2.0, 5.0, 10.0]          # long-scale error (L = 12 km), mGal
SS2 = [0.0, 1.0, 2.0]                # short-scale error (L = 2 km), mGal
STATS = ["r_all", "trend_r", "trend_sd_ratio", "resid_r", "resid_sd_ratio", "lp10_r", "lp10_sd_diff",
         "lp10_sd_ratio", "lp18_r", "lp18_sd_diff", "lp18_sd_ratio"]


def unit_errors(gg, key, i0, n, L, tag):
    """Unit-variance error fields for scenes i0 .. i0+n-1 (seed [key0, key1, tag, scene index])."""
    return np.column_stack([gg.error_field(np.random.default_rng([key[0], key[1], tag, i0 + j]), 1.0, L_err=L)
                            for j in range(n)])


def emulate2(gg, F, key, i0, lam, a, s_long, s_short):
    G = a * gg.lowpass(F.astype(float), lam)
    if s_long > 0:
        G += s_long * unit_errors(gg, key, i0, F.shape[1], 12_000.0, 79)
    if s_short > 0:
        G += s_short * unit_errors(gg, key, i0, F.shape[1], 2_000.0, 80)
    return G


def calib2():
    """Variant family G = a LP_lambda(true) + s_long e_12km + s_short e_2km, scored by how typical
    the real PSG-GGMplus statistics are under each variant (smallest two-sided tail probability
    over the 11 statistics; 50% = all real statistics at the median)."""
    xy, hull, fe, gg = setup()
    z = np.load("results/F_A_test.npz")
    F, D = z["F"].astype(float), z["D"]
    key = KEYS["A_test"][0]
    n = F.shape[1]

    def at_st(G):
        return (gg.Ws @ G, gg.Ws @ gg.lowpass(G, 10_000.0), gg.Ws @ gg.lowpass(G, LAMBDA_EGM))

    sig = {lam: at_st(gg.lowpass(F, lam)) for lam in LAMS2}
    el = at_st(unit_errors(gg, key, 0, n, 12_000.0, 79))
    es = at_st(unit_errors(gg, key, 0, n, 2_000.0, 80))
    real = {h: real_stats(h) for h in ("H1_psg_2670", "H2_psg_2840_converted")}
    rows = []
    for lam in LAMS2:
        for a in A2:
            for sl in SL2:
                for ss in SS2:
                    Gs = [a * sig[lam][k] + sl * el[k] + ss * es[k] for k in range(3)]
                    st = misfit_stats_st(xy, D, *Gs)
                    row = dict(lambda_km=lam / 1e3, a=a, s_long=sl, s_short=ss)
                    for h, tag in (("H1_psg_2670", "H1"), ("H2_psg_2840_converted", "H2")):
                        worst = 50.0
                        for k in STATS:
                            pr = float((st[k] < real[h][k]).mean() * 100)
                            if tag == "H1":
                                row[f"{k}_med"] = float(np.nanmedian(st[k]))
                                row[f"{k}_realpct"] = pr
                            worst = min(worst, pr, 100 - pr)
                        row[f"min_tail_pct_{tag}"] = worst
                    rows.append(row)
    out = pd.DataFrame(rows).sort_values("min_tail_pct_H1", ascending=False)
    out.to_csv("results/ggm_emulation_calibration2.csv", index=False)
    cols = ["lambda_km", "a", "s_long", "s_short", "min_tail_pct_H1", "min_tail_pct_H2"] + [f"{k}_realpct" for k in STATS]
    print(out[cols].head(15).round(1).to_string(index=False))
    print("real (H1):", {k: round(v, 3) for k, v in real["H1_psg_2670"].items()})


def xg_for2(name, gg, lam, a, sl, ss):
    z = np.load(f"results/F_{name}.npz")
    F, D, IDX = z["F"], z["D"], z["idx"]
    key = KEYS[name][0]
    out = []
    for b0 in range(0, F.shape[1], BATCH):
        sl_ = slice(b0, min(b0 + BATCH, F.shape[1]))
        G = emulate2(gg, F[:, sl_], key, b0, lam, a, sl, ss)
        XG = gg.features(G, D[:, sl_])
        out.extend(XG[j, IDX[b0 + j]] for j in range(XG.shape[0]))
    return np.concatenate(out)


def train2(variants):
    """variants: list of (lambda_m, a, s_long, s_short, label)."""
    xy, hull, fe, gg = setup()
    tr, va, te = load("A_train"), load("A_val"), load("A_test")
    T = list(te["targets"])
    rows = []
    for lam, a, sl, ss, label in variants:
        XGs = {n: xg_for2(n, gg, lam, a, sl, ss) for n in KEYS}
        tag = f"L{lam/1e3:g}_a{a:g}_sl{sl:g}_ss{ss:g}"
        for model, use_psg in [("PSG + GGM", True), ("GGM only (regional)", False)]:
            if use_psg:
                Xtr, Xva, Xte = (np.hstack([d["X"], XGs[n]]) for d, n in ((tr, "A_train"), (va, "A_val"), (te, "A_test")))
            else:
                Xtr, Xva, Xte = (XGs[n][:, :3] for n in KEYS)
            m = hgb().fit(Xtr, tr["y"], X_val=Xva, y_val=va["y"])
            p = m.predict_proba(Xte)[:, 1]
            rows.append(dict(variant=label, model=model, lambda_km=lam / 1e3, a=a, s_long=sl, s_short=ss,
                             AUC=roc_auc_score(te["y"], p), AP=average_precision_score(te["y"], p), n_iter=m.n_iter_,
                             **{f"recall@{k}FP": v for k, v in froc(fe, p.reshape(-1, fe.n_in), T).items()}))
            if use_psg:
                joblib.dump(m, f"models/hgb_psg_ggm_{tag}.joblib")
                np.save(f"results/pred_psg_ggm_{tag}_A.npy", p.astype(np.float32))
            print(rows[-1], flush=True)
    path = "results/ggm_value_calibrated.csv"
    try:
        old = pd.read_csv(path)
        out = pd.concat([old[~old.variant.isin([v[4] for v in variants])], pd.DataFrame(rows)], ignore_index=True)
    except FileNotFoundError:
        out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(out.round(3).to_string(index=False))


def xg_for(name, gg, lam, a, e, fe):
    z = np.load(f"results/F_{name}.npz")
    F, D, IDX = z["F"], z["D"], z["idx"]
    key = KEYS[name][0]
    out = []
    for b0 in range(0, F.shape[1], BATCH):
        sl = slice(b0, min(b0 + BATCH, F.shape[1]))
        G = emulate(gg, F[:, sl], key, b0, lam, a, e)
        XG = gg.features(G, D[:, sl])
        out.extend(XG[j, IDX[b0 + j]] for j in range(XG.shape[0]))
    return np.concatenate(out)


def train(lam, a, errs):
    xy, hull, fe, gg = setup()
    tr, va, te = load("A_train"), load("A_val"), load("A_test")
    T = list(te["targets"])
    rows = []
    for e in errs:
        XGs = {n: xg_for(n, gg, lam, a, e, fe) for n in KEYS}
        for label, use_psg in [("PSG + GGM", True), ("GGM only (regional)", False)]:
            if use_psg:
                Xtr, Xva, Xte = (np.hstack([d["X"], XGs[n]]) for d, n in ((tr, "A_train"), (va, "A_val"), (te, "A_test")))
            else:
                Xtr, Xva, Xte = (XGs[n][:, :3] for n in KEYS)
            m = hgb().fit(Xtr, tr["y"], X_val=Xva, y_val=va["y"])
            p = m.predict_proba(Xte)[:, 1]
            rows.append(dict(model=label, lambda_km=lam / 1e3, a=a, ggm_err_mGal=e, AUC=roc_auc_score(te["y"], p),
                             AP=average_precision_score(te["y"], p), n_iter=m.n_iter_,
                             **{f"recall@{k}FP": v for k, v in froc(fe, p.reshape(-1, fe.n_in), T).items()}))
            if use_psg:
                joblib.dump(m, f"models/hgb_psg_ggm_L{lam/1e3:g}_a{a:g}_e{e:g}.joblib")
                np.save(f"results/pred_psg_ggm_L{lam/1e3:g}_a{a:g}_e{e:g}_A.npy", p.astype(np.float32))
            print(rows[-1], flush=True)
    path = "results/ggm_value_realistic.csv"
    try:
        old = pd.read_csv(path)
        keep = ~((old.lambda_km == lam / 1e3) & (old.a == a) & old.ggm_err_mGal.isin(errs))
        out = pd.concat([old[keep], pd.DataFrame(rows)], ignore_index=True)
    except FileNotFoundError:
        out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "regen":
        regen()
    elif step == "calib":
        calib()
    elif step == "calib2":
        calib2()
    elif step == "train2":
        # train2 lam a s_long s_short label [lam a s_long s_short label ...]
        v = sys.argv[2:]
        train2([(float(v[i]), float(v[i + 1]), float(v[i + 2]), float(v[i + 3]), v[i + 4]) for i in range(0, len(v), 5)])
    elif step == "train":
        train(float(sys.argv[2]), float(sys.argv[3]), [float(v) for v in sys.argv[4:]])
