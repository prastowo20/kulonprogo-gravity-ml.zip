"""Build the shared synthetic datasets used by every model (scenario A = reference prior).

The scenes are identical to ML v1 (same seeds): station data agree with the saved v1 test set
to ~1e-12 mGal. In addition, the noise-free field is evaluated on a 1 km grid so that a
GGMplus-like regional product can be emulated at four error levels (0, 2, 5, 10 mGal).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from features import FEATURES, FeatureExtractor
from geometry import convex_hull, load_psg
from ggm import GGM_FEATURES, GGMGrid
from priors_cfg import SCENARIOS
from scenes import ProfileBank, SceneGenerator, cell_labels

SEED = 20260924                     # same as ML v1 train.py
GGM_ERR = [0.0, 2.0, 5.0, 10.0]     # mGal
N_TRAIN, N_VAL, N_TEST, CELLS = 4000, 800, 1000, 300
BATCH = 250


def build(name, n, rng_key, cells, fe, gg, gen, with_ggm):
    rng = np.random.default_rng(rng_key)
    Xp, Xg, ys, sid, T, Dall = [], [], [], [], [], []
    for b0 in range(0, n, BATCH):
        sc = [gen.sample(rng, extra_xy=gg.nodes if with_ggm else None) for _ in range(min(BATCH, n - b0))]
        D = np.column_stack([s.d for s in sc])
        X = fe.transform(D)
        if with_ggm:
            F = gg.lowpass(np.column_stack([s.field_extra for s in sc]))
            XG = []
            for lev, s_err in enumerate(GGM_ERR):
                E = np.column_stack([gg.error_field(np.random.default_rng([rng_key[0], rng_key[1], 77, lev, b0 + j]),
                                                    s_err) for j in range(len(sc))])
                XG.append(gg.features(F + E, D))
            XG = np.concatenate(XG, axis=-1)                 # (B, n_in, 18)
        for j, s in enumerate(sc):
            y = cell_labels(fe.cells, s.targets)
            idx = np.arange(fe.n_in) if cells is None else rng.choice(fe.n_in, cells, replace=False)
            Xp.append(X[j, idx])
            if with_ggm:
                Xg.append(XG[j, idx])
            ys.append(y[idx])
            sid.append(np.full(len(idx), b0 + j))
            T.append(s.targets)
        Dall.append(D)
    out = dict(X=np.concatenate(Xp), y=np.concatenate(ys), sid=np.concatenate(sid).astype(np.int32),
               D=np.hstack(Dall), targets=np.array(T, dtype=object))
    if with_ggm:
        out["XG"] = np.concatenate(Xg)
    np.savez(f"results/ds_{name}.npz", **out)
    return out


def main():
    t0 = time.time()
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    gg = GGMGrid(fe)
    bank = ProfileBank("models/profile_bank.npz")
    genA = SceneGenerator(xy, convex_hull(xy), priors=SCENARIOS["A_clutter"], bank=bank)
    build("A_train", N_TRAIN, [SEED, 0, 0], CELLS, fe, gg, genA, True)
    print(f"train done {time.time() - t0:.0f}s", flush=True)
    build("A_val", N_VAL, [SEED, 0, 900], CELLS, fe, gg, genA, True)
    build("A_test", N_TEST, [SEED, 0, 901], None, fe, gg, genA, True)
    print(f"A done {time.time() - t0:.0f}s", flush=True)
    for si, name in [(1, "B_mixed"), (2, "C_body")]:
        gen = SceneGenerator(xy, convex_hull(xy), priors=SCENARIOS[name], bank=bank)
        build(f"{name[0]}_test", N_TEST, [SEED, si, 901], None, fe, gg, gen, False)
    bank.save()
    print(f"all done {time.time() - t0:.0f}s; features: {FEATURES + [f'{f}_e{e:g}' for e in GGM_ERR for f in GGM_FEATURES]}")


if __name__ == "__main__":
    main()
