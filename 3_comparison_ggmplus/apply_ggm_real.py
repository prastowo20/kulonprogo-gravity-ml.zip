"""Apply the PSG-only model and the PSG + GGM models trained on the calibrated GGMplus emulation
(ggm_emulation.py calib2/train2) to the real data.

Real GGM product: the re-reduced GGMplus CBA (ggm_real.py, all wavelengths, as in the
calibration) interpolated to the 1 km GGM grid nodes. Station data: PSG CBA under the two
reduction-density hypotheses of psg_density.py (H1 as is; H2 converted 2840 -> 2670 kg/m^3).
Global test: 1000 target-free scenes (prior A) with the same GGM emulation variant.
Also a covariate-shift check of the real GGM features against the synthetic test scenes.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from ggm import GGM_FEATURES
from ggm_emulation import emulate2, setup, xg_for2
from priors_cfg import SCENARIOS
from scenes import ProfileBank, SceneGenerator

GAJAH = (401448.0, 9140279.0)
P1, P2 = (400000.0, 9145000.0), (402000.0, 9136000.0)
VARIANTS = [("calibrated_best_fit", 45000.0, 0.33, 10.0, 0.0),
            ("calibrated_best_fit_short", 45000.0, 0.33, 10.0, 2.0),
            ("calibrated_most_optimistic", 45000.0, 0.5, 0.0, 2.0)]
N_NULL = 1000


def tag(lam, a, sl, ss):
    return f"L{lam/1e3:g}_a{a:g}_sl{sl:g}_ss{ss:g}"


def summary(fe, p, null_max, ref=None):
    at = lambda q: float(p[np.argmin(np.hypot(fe.cells[:, 0] - q[0], fe.cells[:, 1] - q[1]))])
    out = dict(max_p=float(p.max()), argmax=[float(v) for v in fe.cells[p.argmax()]], p_gajah=at(GAJAH),
               p_P1=at(P1), p_P2=at(P2), area_p_ge_0p3_km2=float((p >= 0.3).sum() * 0.25),
               global_p=float((null_max >= p.max()).mean()))
    if ref is not None:
        out["corr_with_psg_only_map"] = float(np.corrcoef(p, ref)[0, 1])
    return out


def main():
    xy, hull, fe, gg = setup()
    red = np.load("results/ggm_real_reduced.npz")
    pts = np.column_stack([red["x"], red["y"]])
    G = griddata(pts, red["cba"], gg.nodes, method="linear")
    miss = np.isnan(G)
    G[miss] = griddata(pts, red["cba"], gg.nodes[miss], method="nearest")
    dens = pd.read_csv("data/derived/psg_cba_H1_H2.csv")
    assert np.allclose(dens[["X", "Y"]].to_numpy(), xy)
    Dsets = {"H1": dens.CBA_H1.to_numpy(), "H2": dens.CBA_H2_2670.to_numpy()}

    gen = SceneGenerator(xy, hull, priors=dict(SCENARIOS["A_clutter"], n_targets_p=[1.0, 0, 0, 0]),
                         bank=ProfileBank("models/profile_bank.npz"))
    rng = np.random.default_rng([4242, 0])
    sc = [gen.sample(rng, extra_xy=gg.nodes) for _ in range(N_NULL)]
    Dn = np.column_stack([s.d for s in sc])
    Fn = np.column_stack([s.field_extra for s in sc])
    Xn = fe.transform(Dn)
    base = joblib.load("models/hgb_A_clutter_m0.joblib")
    null_base = base.predict_proba(Xn.reshape(-1, Xn.shape[2]))[:, 1].reshape(N_NULL, -1).max(1)

    out = {"GGM grid nodes filled by nearest (outside GGMplus tile)": int(miss.sum())}
    maps = {}
    for h, d in Dsets.items():
        Xp = fe.transform(d)[0]
        pb = base.predict_proba(Xp)[:, 1]
        out[f"PSG only | {h}"] = summary(fe, pb, null_base)
        maps[f"PSG_only_{h}"] = pb
        XG = gg.features(G[:, None], d[:, None])[0]
        for name, lam, a, sl, ss in VARIANTS:
            m = joblib.load(f"models/hgb_psg_ggm_{tag(lam, a, sl, ss)}.joblib")
            p = m.predict_proba(np.hstack([Xp, XG]))[:, 1]
            Gn = emulate2(gg, Fn, [4242, 1], 0, lam, a, sl, ss)
            XGn = gg.features(Gn, Dn)
            pn = m.predict_proba(np.concatenate([Xn, XGn], axis=-1).reshape(-1, Xn.shape[2] + XGn.shape[2]))[:, 1]
            out[f"PSG + GGM ({name}) | {h}"] = summary(fe, p, pn.reshape(N_NULL, -1).max(1), ref=pb)
            maps[f"PSG_GGM_{name}_{h}"] = p
            print(f"{name} | {h}:", out[f"PSG + GGM ({name}) | {h}"], flush=True)

    # covariate shift of the real GGM features (H1) vs synthetic test scenes of each variant
    XG1 = gg.features(G[:, None], Dsets["H1"][:, None])[0]
    shift = {}
    for name, lam, a, sl, ss in VARIANTS:
        S = xg_for2("A_test", gg, lam, a, sl, ss)
        lo, hi = np.percentile(S, 0.5, axis=0), np.percentile(S, 99.5, axis=0)
        shift[name] = {f: float(((XG1[:, k] < lo[k]) | (XG1[:, k] > hi[k])).mean()) for k, f in enumerate(GGM_FEATURES)}
    out["fraction_of_real_cells_outside_synthetic_0.5-99.5pct"] = shift
    np.savez_compressed("results/real_ggm_maps.npz", cells=fe.cells, **maps)
    json.dump(out, open("results/real_ggm_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
