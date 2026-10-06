"""Prior predictive check: does the synthetic scene generator produce data that look like
the real PSG data? Compares summary statistics of plane-removed station residuals."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from detectors import plane_projector
from geometry import convex_hull, load_psg, pairwise_dist
from scenes import ProfileBank, SceneGenerator

LAGS = [(0, 3000), (3000, 6000), (6000, 10000), (10000, 15000)]


def stats(r, D):
    iu = np.triu_indices(len(r), 1)
    d = D[iu]
    g = 0.5 * (r[:, None] - r[None, :])[iu] ** 2
    out = dict(std=r.std(), max=r.max(), min=r.min(), range=r.max() - r.min(),
               skew=((r - r.mean()) ** 3).mean() / r.std() ** 3)
    for lo, hi in LAGS:
        m = (d >= lo) & (d < hi)
        out[f"gamma_{lo // 1000}_{hi // 1000}km"] = g[m].mean()
    out["gamma_ratio_short_long"] = out["gamma_0_3km"] / out["gamma_10_15km"]
    return out


def run(n=2000, seed=777, priors=None, tag="", bank=None):
    xy, df = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    P = plane_projector(xy)
    D = pairwise_dist(xy, xy)
    real = stats(P @ df.CBA.to_numpy(), D)
    bank = bank or ProfileBank("models/profile_bank.npz")
    gen = SceneGenerator(xy, hull, priors=priors, bank=bank)
    rng = np.random.default_rng(seed)
    syn = pd.DataFrame([stats(P @ gen.sample(rng).d, D) for _ in range(n)])
    bank.save()
    rows = []
    for k, v in real.items():
        rows.append(dict(statistic=k, real=v, syn_p05=syn[k].quantile(0.05), syn_p50=syn[k].median(),
                         syn_p95=syn[k].quantile(0.95), real_percentile=100 * (syn[k] < v).mean()))
    t = pd.DataFrame(rows)
    t.to_csv(f"results/sim2real{tag}.csv", index=False)
    syn.to_csv(f"results/sim2real_samples{tag}.csv", index=False)
    return t


if __name__ == "__main__":
    # reproduces results/sim2real_<scenario>.csv and results/prior_predictive_percentiles.csv
    from priors_cfg import SCENARIOS
    pd.set_option("display.width", 160)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    bank = ProfileBank("models/profile_bank.npz")
    summ = {}
    for k, pr in SCENARIOS.items():
        t = run(n=n, seed=781, priors=pr, tag=f"_{k}", bank=bank)
        summ[k] = t.set_index("statistic").real_percentile
    out = pd.DataFrame(summ)
    out.loc["mean_abs_dev_from_50"] = (out - 50).abs().mean()
    out.round(2).to_csv("results/prior_predictive_percentiles.csv")
    print(out.round(1).to_string())
