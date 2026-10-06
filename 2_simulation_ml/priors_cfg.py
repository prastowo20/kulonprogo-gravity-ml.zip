"""Prior scenarios for the synthetic generator (overrides of scenes.PRIORS).

All three pass (or nearly pass) the prior predictive check: the real PSG residual statistics fall
inside the synthetic 5-95% band. They attribute the observed variance differently - to smooth
geological clutter (A) or to discrete bodies (C) - which gravity alone cannot resolve, so the
spread between them is reported as prior uncertainty. A fits the real statistics best and is
the reference scenario.
"""
MAIN = dict(grf_std=(8.0, 18.0), grf_len=(4000.0, 14000.0), neg_drho=(200.0, 450.0), n_neg_mean=1.0,
            target_drho=(150.0, 450.0))
CLUTTER = dict(MAIN, grf_std=(12.0, 22.0), target_drho=(150.0, 350.0))
BODY = dict(MAIN, grf_std=(6.0, 14.0), n_targets_p=[0.10, 0.30, 0.35, 0.25], target_drho=(250.0, 500.0),
            n_neg_mean=1.5, neg_drho=(250.0, 500.0), neg_R=(2000, 7000))
SCENARIOS = {"A_clutter": CLUTTER, "B_mixed": MAIN, "C_body": BODY}
REFERENCE = "A_clutter"
