#!/bin/bash
# Full ML v2 pipeline, in order (2 CPU cores: ~1.5-2 h in total).
# Needs data/gravity_requested_area.csv (terrestrial CBA; restricted, or the synthetic demo file),
# data/ggmplus_oasis.xls (GGMplus tile) and the HGB models of stage 2 copied into models/:
#   mkdir -p models && cp ../2_simulation_ml/models/hgb_*.joblib ../2_simulation_ml/models/profile_bank.npz models/
set -e
python3 test_cylinder.py                                   # forward-model checks
python3 datasets.py                                        # shared synthetic datasets (seeds fixed)
python3 compare_models.py                                  # 9 methods, same protocol
python3 bootstrap_ci.py                                    # per-model 95% CIs
python3 paired_diff.py                                     # paired differences vs HGB
python3 leakage_demo.py                                    # random vs scene-wise split
python3 ggm_real.py data/ggmplus_oasis.xls                 # GGMplus re-reduction + RQ1 comparison
python3 psg_density.py data/ggmplus_oasis.xls              # PSG reduction-density hypotheses H1/H2
python3 ggm_value.py                                       # naive GGM emulation (LP 10 km)
python3 ggm_emulation.py regen                             # node fields of the synthetic scenes
python3 ggm_emulation.py calib                             # stage-1 misfit check
python3 ggm_emulation.py calib2                            # stage-2 misfit check (144 variants)
python3 ggm_emulation.py train2 18300 1 0 0 EGM2008-limit_perfect 45000 0.5 0 2 calibrated_most_optimistic \
    45000 0.33 10 0 calibrated_best_fit 45000 0.33 10 2 calibrated_best_fit_short   # value of GGMplus features
python3 apply_ggm_real.py                                  # real data: terrestrial only vs terrestrial + GGMplus
python3 ensemble_density.py                                # reference ensemble under H1/H2 + global tests
python3 null_benchmark.py                                  # null distributions (Fig. 10d-f) + hotspot uncertainty (Table 7)
python3 paired_diff.py --ggm                               # paired differences for the GGMplus variants
python3 plots_ms.py                                        # manuscript figures (FIG_OUT, default ../figures_manuscript)
