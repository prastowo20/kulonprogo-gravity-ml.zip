# kulonprogo-gravity-ml

Code for the article

> Prastowo, R., Sukiyah, E., Syafri, I., Yuniardi, Y. *Locating the eruption centres of deeply eroded volcanoes
> from sparse gravity data: a simulation-trained machine-learning test in the Kulon Progo dome, Java, Indonesia.*
> Submitted to *Volcanica*.

The code trains machine-learning (ML) detectors of eroded eruption-centre zones (dense intrusive cores,
radius 1.5–4 km) on synthetic gravity scenes that are sampled at the geometry of a real, sparse terrestrial
station network. It then evaluates the detectors on unseen scenes and applies them to the real data. It also
re-reduces the GGMplus global gravity model, compares it with the terrestrial data band by band, and measures
what GGMplus-derived features add when the GGMplus emulation is calibrated against the observed misfit.

All random seeds are fixed. Every number in the article is read from the result files written by these
scripts (see [Where the paper's numbers come from](#where-the-papers-numbers-come-from)).

## Contents

| Folder | Paper section | What it does |
|---|---|---|
| `1_detectability/` | 3.3, 4.2; Fig. 4; Figs. S1–S2; Table 4 | Monte Carlo probability of detection of vertical cylinders for the actual network and for regular grids. Detectors: D1 matched filter (true shape known), D2 map reading; familywise false-alarm rate 5%. |
| `2_simulation_ml/` | 3.4–3.8, 3.11, 4.3, 4.7; Figs. 5, 6, 10; Figs. S3–S4; Tables 1, S2 | Scene generator, prior predictive check (`sim2real.py`), features, gradient-boosting ensemble (`train.py`), FROC and conventional indicators (`evaluate.py`), benchmark against D1/D2 (`benchmark.py`), permutation importance, real-data maps with uncertainty (`apply_real.py`), global test (`global_test.py`), comparison with the mapped geology (`geology_vs_probability.py`). |
| `3_comparison_ggmplus/` | 3.2, 3.7–3.11, 4.1, 4.4–4.7; Figs. 3, 7–9; Figs. S5–S6; Tables 3, 5–7, S4–S5 | Nine learners under one protocol (`compare_models.py`), bootstrap and paired intervals, leakage experiment, GGMplus re-reduction with prisms (`ggm_real.py`), reduction-density hypotheses (`psg_density.py`), GGMplus emulation and misfit calibration (`ggm_emulation.py`), value of GGMplus features, real-data application. |
| `4_manuscript_figures/` | Figs. 1–2; graphical abstract | Study-area map (original and with the geology panel, `make_fig01_geology.py`), workflow diagram and graphical abstract. |
| `demo/` | – | Generator of a synthetic demonstration station file (the terrestrial data are restricted). |

In the code, "PSG" (Pusat Survei Geologi) denotes the terrestrial gravity data set, and "GGM" denotes GGMplus.
H1 and H2 are the two reduction-density hypotheses of the terrestrial complete Bouguer anomaly (CBA): H1 uses
the CBA as supplied; H2 converts it from 2840 to 2670 kg/m³ (Eq. 2 of the paper).

## Installation

Python 3.11 is required. The versions used for the paper are pinned in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r requirements.txt
```

Hardware: any 64-bit computer with at least 8 GB of RAM. The complete pipeline ran in about two hours on two
CPU cores (stage 1 ≈ 3 min, stage 2 ≈ 25 min, stage 3 ≈ 1.5 h).

## Input data

| File (per stage folder) | Content | Availability |
|---|---|---|
| `data/gravity_requested_area.csv` | Terrestrial stations. Columns `X`, `Y` (UTM zone 49S, m), `LATITUDE`, `LONGITUDE`, `CBA` (mGal). | Restricted: Center for Geological Survey, Geological Agency of Indonesia (on request). Use the demo file instead (below). |
| `3_comparison_ggmplus/data/ggmplus_oasis.xls` | GGMplus tile exported as a spreadsheet: one header row, then columns x, y (UTM 49S, m), surface height z (m), gravity g (mGal), row by row on a regular grid. | GGMplus is public: http://ddfe.curtin.edu.au/gravitymodels/GGMplus/ (Hirt et al., 2013). |
| folder given by the environment variable `GEOLOGY_DIR` | `Geologi.shp` (unit polygons, code `SANDI`) and `faultline.shp` of the digital geological map of Indonesia (WGS84). Needed only for Fig. 1a, the outlines in Fig. 10a (`--volcanica`) and the geology comparison of Section 4.7. | Pusat Penelitian dan Pengembangan Geologi (Geological Agency of Indonesia); not redistributed. Without it, the geology steps stop with a message and Fig. 10 is drawn without outlines. |

The data files are excluded from version control (`.gitignore`).

## Quick start with synthetic demonstration data

```bash
python demo/make_demo_data.py            # writes a SYNTHETIC station file into the three stage folders
cd 1_detectability
python test_cylinder.py                  # forward-model checks (analytic limits, brute force)
python run_pod.py && python postprocess.py && python verify.py
cd ../2_simulation_ml
python sim2real.py 200                   # prior predictive check with 200 scenes per prior (≈1 min)
python train.py                          # generator, features, HGB ensemble, test metrics (≈12 min)
python evaluate.py && python apply_real.py && python global_test.py
```

The results of the demonstration are not those of the paper. They show that the pipeline runs end to end and
how its outputs look.

## Full reproduction (with the real input files)

```bash
cd 1_detectability
python test_cylinder.py && python run_pod.py && python postprocess.py && python verify.py && python plots_ms.py

cd ../2_simulation_ml
python test_cylinder.py && python sim2real.py && python train.py && python evaluate.py && python benchmark.py
python importance.py && python apply_real.py && python global_test.py && python plots_ml_ms.py

cd ../3_comparison_ggmplus
mkdir -p models && cp ../2_simulation_ml/models/hgb_*.joblib ../2_simulation_ml/models/profile_bank.npz models/
bash run_all.sh                          # stage 3, all steps in order (≈1.5 h)

cd ../4_manuscript_figures && python make_new_figs.py && python make_graphical_abstract.py

# geology (revision of 5 October 2026); needs the geological map, see Input data
export GEOLOGY_DIR=/path/to/geological_map
cd ../2_simulation_ml && python geology_vs_probability.py && python plots_fig10_null.py --volcanica
cd ../4_manuscript_figures && python make_fig01_geology.py
```

Figures are written to `figures_manuscript/` at the repository root (set `FIG_OUT` to change this). Each
stage also writes its own `results/` folder (CSV/JSON/NPZ) and `models/` folder (joblib).

## Main options

- **Scene priors:** `priors_cfg.py` defines priors A (clutter-rich, reference), B and C. They override the
  defaults in `scenes.py` (`PRIORS`). Change them there to explore other geological assumptions.
- **Target definition:** radius, depth, height and density ranges are in `PRIORS` (`target_*`).
- **Features:** rings, annuli and matched-filter radii are set at the top of `features.py`.
- **Learners and tuning grids:** `compare_models.py` (`fit_*` functions); ensemble settings in
  `2_simulation_ml/train.py`.
- **GGMplus emulation family:** `ggm_emulation.py` (`LAMS2`, `A2`, `SL2`, `SS2`). Train a variant with
  `python ggm_emulation.py train2 <lambda_m> <a> <s_long> <s_short> <label>`.
- **Reduction-density hypotheses:** `psg_density.py`.

## Where the paper's numbers come from

| Paper item | Result file |
|---|---|
| Table 3, Section 4.1 | `3_comparison_ggmplus/results/psg_density.json`, `ggm_real_comparison.json` |
| Fig. 4, Table 4 | `1_detectability/results/R80_table.csv`, `verification_log.txt` |
| Section 4.3, Figs. 5–6, Table S2 | `2_simulation_ml/results/prior_predictive_percentiles.csv`, `ml_metrics.csv`, `froc_baselines_*.csv`, `ml_object_summary.csv`, `importance_grouped.csv` |
| Table S4, Fig. 7 | `3_comparison_ggmplus/results/model_comparison.csv`, `model_comparison_ci.csv`, `model_comparison_paired.csv` |
| Fig. 8 | `3_comparison_ggmplus/results/leakage_summary.json` |
| Table 5, Fig. 9, Table S5 | `ggm_value.csv`, `ggm_value_calibrated.csv`, `ggm_value_paired.csv`, `ggm_emulation_calibration2.csv` |
| Tables 6–7, Section 4.7, Fig. 10 | `2_simulation_ml/results/real_summary.json`, `real_peaks.csv`, `global_test.json`; `3_comparison_ggmplus/results/ensemble_density.json`, `real_ggm_summary.json`, `null_benchmark.json` |
| Sections 2.1 and 4.7 (comparison with the mapped geology), Fig. 1a | `2_simulation_ml/results/geology_vs_probability.json` |

## Revision additions (30 September and 1 October 2026)

Added for the revised version of the article (feedback: make the detection limit the main result, document
the simulation completely, and always pair probability maps with a null benchmark). The same revision is used
in the English (*Volcanica*) and the Indonesian versions of the manuscript:

| Script | Output | Purpose |
|---|---|---|
| `3_comparison_ggmplus/null_benchmark.py` | `results/null_benchmark.npz`, `results/null_benchmark.json` | Regenerates the maximum probability of the map in 1000 target-free scenes per prior (seeds `[4242, prior index]`, as `global_test.py`), checks the stored medians, 95th percentiles and global p-values, and recomputes the station jackknife at P1, P2 and the Gajah summit. Run after `ensemble_density.py` and stage 2. |
| `2_simulation_ml/plots_fig10_null.py` | `fig10_real_map_null_uncertainty.{png,pdf}`; with `--volcanica`: `fig10_real_map_null_uncertainty_volcanica.{png,pdf}` | Figure 10 with a null-benchmark row (real map maximum, null median, 95th percentile, global p) between the probability maps and the uncertainty maps. The `--volcanica` option draws the same figure on a smaller canvas (173.7 × 196.7 mm) that fits the *Volcanica* limit of 175 × 200 mm. |
| `4_manuscript_figures/make_fig02_rev.py` | `fig02_workflow_rev.{png,pdf}`; with `--volcanica`: `fig02_workflow_rev_volcanica.{png,pdf}` | Workflow diagram with the detectability limit as the first step, the validation gate and the null benchmark outlined. The `--volcanica` option uses SI units (kg m⁻³) and the wording of the English manuscript. |

No existing result changes: `null_benchmark.py` reproduces the stored global-test summaries exactly and stops
the figure script if they differ.

## Revision additions (5 October 2026): comparison with the mapped geology

The model never uses geology. These scripts compare the terrestrial data and the reference probability map with
the digital geological map of Indonesia, which is not redistributed (set `GEOLOGY_DIR`, see Input data):

| Script | Output | Purpose |
|---|---|---|
| `2_simulation_ml/geology_vs_probability.py` | `results/geology_vs_probability.json` | Unit shares inside the station hull; mean p and cells with p ≥ 0.3 per unit; units at P1, P2 and the Gajah summit; mean CBA of the stations on and off the andesite intrusions after a least-squares plane (H1 and H2, leave-one-station-out range); 1600 translations of the geological map (every offset of 2–6 km on a 250 m lattice) as the reference distribution; unit elevations and Spearman correlation of p with elevation (GGMplus surface heights). Needs `3_comparison_ggmplus/data/derived/psg_cba_H1_H2.csv` from `psg_density.py`. |
| `4_manuscript_figures/make_fig01_geology.py` | `fig01_study_area_geology.{png,pdf}` | Figure 1 with (a) the geology of the data window and (b) the study area and terrestrial CBA (174.4 × 118.2 mm). |
| `2_simulation_ml/plots_fig10_null.py --volcanica` | `fig10_real_map_null_uncertainty_volcanica.{png,pdf}` | Panel (a) now also outlines the mapped andesite intrusions and dacite inside the station hull. |

## Revision additions (6 October 2026): figures in the Volcanica house style

The Volcanica Word template asks for panels labelled with upper-case letters in a 12-point sans-serif font. The
figure scripts draw this version when `PANEL_STYLE=volcanica` is set (helper `volcanica_figstyle.py` at the
repository root): leading panel tokens such as "(a) " or "a  " in the axes titles become bold upper-case letters,
whose size is corrected for the later reduction to the 85 or 175 mm column width; panels without letters before
(Fig. 4, Figs. S2, S3, S5) receive them; density units in figure labels are given in kg m⁻³; the note of Fig. 8B
moves below the axis; the top-row titles of Fig. 10 put the statistics on two short lines. The figures are written
to `$FIG_OUT/volcanica` (default `figures_manuscript/volcanica`); without the variable, the scripts write the
earlier lower-case version unchanged.

```bash
export PANEL_STYLE=volcanica GEOLOGY_DIR=/path/to/geological_map
cd 1_detectability && python plots_ms.py
cd ../2_simulation_ml && python plots_ml_ms.py bench synth prior imp && python plots_fig10_null.py --volcanica
cd ../3_comparison_ggmplus && python plots_ms.py
cd ../4_manuscript_figures && python make_fig01_geology.py && python make_fig02_rev.py --volcanica
```

## Licence and citation

MIT licence (see `LICENSE`). Please cite the article above and this repository (`CITATION.cff`).
