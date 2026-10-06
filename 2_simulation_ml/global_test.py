"""Global test: how often does a map as peaked as the real one arise in scenes WITHOUT any
eruption-centre target (same prior, same geometry)? Also a station-permutation null."""
from __future__ import annotations

import json

import joblib
import numpy as np

from apply_real import predict
from features import FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import REFERENCE, SCENARIOS
from scenes import ProfileBank, SceneGenerator

N = 1000


def main():
    xy, df = load_psg("data/gravity_requested_area.csv")
    d = df.CBA.to_numpy()
    hull = convex_hull(xy)
    fe = FeatureExtractor(xy, hull)
    bank = ProfileBank("models/profile_bank.npz")
    out = {}
    for si, (name, pr) in enumerate(SCENARIOS.items()):
        models = ([joblib.load(f"models/hgb_{name}_m{m}.joblib") for m in range(5)] if name == REFERENCE
                  else [joblib.load(f"models/hgb_{name}_m0.joblib")])
        real_max = float(predict(models, fe.transform(d)).mean(0).max())
        gen = SceneGenerator(xy, hull, priors=dict(pr, n_targets_p=[1.0, 0.0, 0.0, 0.0]), bank=bank)
        rng = np.random.default_rng([4242, si])
        D0 = np.column_stack([gen.sample(rng).d for _ in range(N)])
        mx0 = np.concatenate([predict(models, fe.transform(D0[:, i:i + 250])).mean(0).max(1) for i in range(0, N, 250)])
        Dp = np.column_stack([rng.permutation(d) for _ in range(N)])
        mxp = np.concatenate([predict(models, fe.transform(Dp[:, i:i + 250])).mean(0).max(1) for i in range(0, N, 250)])
        out[name] = dict(real_max_p=real_max, frac_no_target_scenes_ge_real=float((mx0 >= real_max).mean()),
                         no_target_max_p_median=float(np.median(mx0)), no_target_max_p_p95=float(np.quantile(mx0, 0.95)),
                         frac_permuted_ge_real=float((mxp >= real_max).mean()),
                         permuted_max_p_median=float(np.median(mxp)))
        print(name, json.dumps(out[name]), flush=True)
    bank.save()
    json.dump(out, open("results/global_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
