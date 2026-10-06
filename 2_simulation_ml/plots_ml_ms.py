"""Figures for ML v1 (PNG 300 dpi + PDF)."""
from __future__ import annotations

import json
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from features import FeatureExtractor  # noqa: E402
from geometry import convex_hull, load_psg  # noqa: E402
from priors_cfg import REFERENCE, SCENARIOS  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, "..")  # repository root: volcanica_figstyle.py
import volcanica_figstyle as VS  # noqa: E402  (PANEL_STYLE=volcanica: upper-case 12 pt panel letters)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]                  # categorical slots 1-3 (all-pairs safe)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
SCEN_LABEL = {"A_clutter": "A: clutter-rich (reference)", "B_mixed": "B: mixed", "C_body": "C: body-rich"}
GAJAH = (401.448, 9140.279)

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
})


NAMES = {'fig5_real_probability_maps': 'fig10_real_map_uncertainty', 'fig6_benchmark_ML_vs_D1_D2': 'figS3_benchmark_detectors', 'fig7_synthetic_detection': 'fig06_froc_indicators', 'fig8_calibration_prior_check': 'fig05_prior_check_reliability', 'fig9_feature_importance': 'figS4_feature_importance'}
import os
FIG_OUT = os.environ.get("FIG_OUT", "../figures_manuscript")
os.makedirs(FIG_OUT, exist_ok=True)


def save(fig, name):
    name = NAMES.get(name, name)
    out = FIG_OUT
    if VS.VOLC:
        VS.relabel(fig)
        out = VS.out_dir(out)
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def map_panel(ax, fe, values, levels, title, xy=None, peaks=None, show_gajah=True, label_gajah=False):
    ny, nx = fe.shape
    Z = np.full(ny * nx, np.nan)
    Z[fe.inside] = values
    Z = Z.reshape(ny, nx)
    cmap = ListedColormap(SEQ[: len(levels) - 1])
    cf = ax.pcolormesh(fe.xs / 1e3, fe.ys / 1e3, Z, cmap=cmap, norm=BoundaryNorm(levels, cmap.N), shading="nearest")
    h = np.vstack([fe.hull, fe.hull[:1]]) / 1e3
    ax.plot(h[:, 0], h[:, 1], color=INK2, lw=0.6)
    if xy is not None:
        ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, s=5, c=INK, edgecolors="white", linewidths=0.4, zorder=4)
    if peaks is not None:
        for k, r in enumerate(peaks.itertuples(), 1):
            ax.scatter(r.x / 1e3, r.y / 1e3, marker="^", s=30, facecolors="white", edgecolors=INK, linewidths=0.9,
                       zorder=5)
            ax.annotate(f"P{k}", (r.x / 1e3, r.y / 1e3), xytext=(4, 2), textcoords="offset points", fontsize=6.5,
                        color=INK, zorder=6)
    if show_gajah:
        ax.scatter(*GAJAH, marker="*", s=46, c=CAT[1], edgecolors="white", linewidths=0.5, zorder=6)
        if label_gajah:
            ax.annotate("Gajah summit", GAJAH, xytext=(5, -3), textcoords="offset points", fontsize=6.3, color=INK,
                        va="top", zorder=6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(390, 411)
    ax.set_ylim(9129.5, 9151)
    ax.set_xticks([390, 395, 400, 405, 410])
    ax.set_yticks([9130, 9135, 9140, 9145, 9150])
    ax.set_title(title, loc="left", color=INK)
    ax.ticklabel_format(useOffset=False, style="plain")
    return cf


def row_colorbar(fig, cf, axes, label):
    cb = fig.colorbar(cf, ax=list(axes), orientation="horizontal", fraction=0.04, pad=0.1, aspect=45, shrink=0.6)
    cb.set_label(label, color=INK2, fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5)


def fig_real_maps():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    z = np.load("results/real_maps.npz")
    pk = pd.read_csv("results/real_peaks.csv")
    pk = pk[pk.p >= 0.3]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 6.4), layout="constrained")
    lv_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    for ax, name in zip(axes[0], SCENARIOS):
        v = z["pA"] if name == REFERENCE else z[f"p_{name}"]
        cf = map_panel(ax, fe, v, lv_p, SCEN_LABEL[name], xy=xy, peaks=pk if name == REFERENCE else None,
                       label_gajah=name == REFERENCE)
    row_colorbar(fig, cf, axes[0], "Probability of a complex-scale centre zone")
    lv_u = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
    map_panel(axes[1, 0], fe, z["prior_range"], lv_u, "Prior spread (max − min, A–C)")
    map_panel(axes[1, 1], fe, z["noise_sd"], lv_u, "Data-noise SD (σ = 2 mGal)", xy=xy)
    cf = map_panel(axes[1, 2], fe, z["jk_se"], lv_u, "Station-jackknife SE", xy=xy)
    row_colorbar(fig, cf, axes[1], "Uncertainty (probability units)")
    for ax in axes[1]:
        ax.set_xlabel("Easting (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Northing (km)")
    save(fig, "fig5_real_probability_maps")


def fig_benchmark():
    b = pd.read_csv("results/benchmark_ML_vs_D1_D2.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, sigma in enumerate([2.0, 5.5]):
        for j, zt in enumerate([100, 500, 1000]):
            ax = axes[i, j]
            s = b[(b.sigma == sigma) & (b.zt_m == zt)].sort_values("R_m")
            ax.axhline(0.8, color=INK2, lw=0.7)
            ax.plot(s.R_m / 1e3, s.pod_D1, color=INK, lw=1.2, marker="s", ms=2.8, label="D1 matched filter (true shape)")
            ax.plot(s.R_m / 1e3, s.pod_D2, color=MUTED, lw=1.2, marker="o", ms=2.8, label="D2 map reading")
            ax.plot(s.R_m / 1e3, s.pod_ML, color=CAT[0], lw=1.8, marker="o", ms=3.2, markeredgecolor="white",
                    markeredgewidth=0.5, label="ML (simulation-trained, shape unknown)")
            ax.set_xscale("log")
            ax.set_xticks([0.25, 0.5, 1, 2, 4])
            ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{VS.tok(3 * i + j)}σ = {sigma:g} mGal · top at {zt / 1e3:g} km", loc="left", color=INK)
            if j == 0:
                ax.set_ylabel("Probability of detection")
            if i == 1:
                ax.set_xlabel("Body radius R (km)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig6_benchmark_ML_vs_D1_D2")


METHODS = [("ML (HGB ensemble)", "ML", "#2a78d6"), ("map value (TPS)", "map", "#eb6834"),
           ("annulus contrast 3-6 km", "annulus", "#1baf7a"), ("matched filter R = 2.5 km", "matched", "#eda100"),
           ("-Laplacian (SVD proxy)", "negLaplacian", "#e87ba4")]


def fig_synthetic_performance():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    for lab, key, c in METHODS:
        z = np.load(f"results/froc_curve_{REFERENCE}_{key}.npz")
        o = np.argsort(z["fp"])
        ax.plot(z["fp"][o], z["recall"][o], color=c, lw=1.8 if key == "ML" else 1.2, label=lab,
                zorder=3 if key == "ML" else 2)
    ax.axvline(1.0, color=INK2, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 10)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_ylim(0, 1)
    ax.set_xlabel("False peaks per scene")
    ax.set_ylabel("Recall of centre zones")
    ax.set_title("a  Free-response ROC, prior A (reference)", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.6)
    ax = axes[1]
    xs = np.arange(3)
    for lab, key, c in METHODS:
        vals = []
        for name in SCENARIOS:
            b = pd.read_csv(f"results/froc_baselines_{name}.csv")
            vals.append(b[(b.method == lab) & (b.max_false_peaks_per_scene == 1.0)].recall.iloc[0])
        ax.plot(xs, vals, color=c, lw=1.8 if key == "ML" else 1.2, marker="o", ms=3.5, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3 if key == "ML" else 2)
        if key == "ML":
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 5), textcoords="offset points", ha="center",
                            fontsize=6.3, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["A clutter-rich", "B mixed", "C body-rich"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Recall at ≤ 1 false peak per scene")
    ax.set_title("b  Across priors", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig7_synthetic_detection")


def fig_prior_check():
    lags = [("gamma_0_3km", 1.5), ("gamma_3_6km", 4.5), ("gamma_6_10km", 8.0), ("gamma_10_15km", 12.5)]
    rel = pd.read_csv("results/ml_reliability.csv")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
    for c, name in zip(CAT, SCENARIOS):
        r = rel[(rel.scenario == name) & (rel.n >= 200)]
        ax.plot(r.p_mean, r.obs_freq, color=c, lw=1.4, marker="o", ms=3, markeredgecolor="white",
                markeredgewidth=0.4, label=name[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("a  Reliability", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=6.3, handlelength=1.4)
    real = pd.read_csv("results/sim2real_A_clutter.csv").set_index("statistic").real
    for k, (ax, c, name) in enumerate(zip(axes[1:], CAT, SCENARIOS)):
        s = pd.read_csv(f"results/sim2real_samples_{name}.csv")
        x = [v for _, v in lags]
        ax.fill_between(x, [s[q].quantile(0.05) for q, _ in lags], [s[q].quantile(0.95) for q, _ in lags],
                        color=c, alpha=0.15, lw=0)
        ax.plot(x, [s[q].median() for q, _ in lags], color=c, lw=1.4, label="synthetic median, 5–95%")
        ax.plot(x, [real[q] for q, _ in lags], color=INK, lw=0, marker="D", ms=3.5, label="terrestrial data")
        ax.set_yscale("log")
        ax.set_ylim(3, 400)
        ax.set_yticks([5, 10, 20, 50, 100, 200])
        ax.set_yticklabels(["5", "10", "20", "50", "100", "200"])
        ax.minorticks_off()
        ax.set_xlabel("Lag (km)")
        ax.set_title(f"{'bcd'[k]}  Prior check {name[0]}", loc="left", color=INK)
        if k:
            ax.set_yticklabels([])
    axes[1].set_ylabel("Semivariance (mGal²)")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=6.3, bbox_to_anchor=(1.0, 1.08))
    fig.tight_layout()
    save(fig, "fig8_calibration_prior_check")


def fig_importance():
    imp = pd.read_csv("results/importance_grouped.csv").sort_values("drop_AP_mean")
    fig, ax = plt.subplots(figsize=(3.54, 2.3))
    y = np.arange(len(imp))
    ax.barh(y, imp.drop_AP_mean, xerr=imp.drop_AP_sd, height=0.55, color=CAT[0], error_kw=dict(lw=0.6, ecolor=INK2))
    ax.set_axisbelow(True)
    ax.set_yticks(y)
    ax.set_yticklabels(imp.group, fontsize=6.5)
    ax.set_xlabel("Drop in average precision when permuted")
    ax.grid(axis="y", visible=False)
    ax.set_title("Grouped permutation importance (prior A)", loc="left", color=INK)
    save(fig, "fig9_feature_importance")


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or ["real", "bench", "synth", "prior", "imp"]
    fns = dict(real=fig_real_maps, bench=fig_benchmark, synth=fig_synthetic_performance, prior=fig_prior_check,
               imp=fig_importance)
    for t in todo:
        fns[t]()
    print("figures:", todo)
