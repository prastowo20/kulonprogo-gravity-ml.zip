"""Figures for ML v2: method comparison, real-data maps per method, leakage demo, GGMplus value."""
from __future__ import annotations

import glob
import json
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from features import FeatureExtractor  # noqa: E402
from geometry import convex_hull, load_psg  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, "..")  # repository root: volcanica_figstyle.py
import volcanica_figstyle as VS  # noqa: E402  (PANEL_STYLE=volcanica: upper-case 12 pt panel letters)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
GAJAH = (401.448, 9140.279)
P1, P2 = (400.0, 9145.0), (402.0, 9136.0)
ORDER = ["HGB", "XGB", "RF", "ET", "MLP", "SVM", "LR", "DT", "GMM"]
SHORT = {"LR": "Logistic regression", "DT": "Decision tree", "RF": "Random forest", "ET": "Extra trees",
         "HGB": "Gradient boosting (HGB)", "XGB": "XGBoost", "SVM": "SVM (RBF)", "MLP": "Neural network (MLP)",
         "GMM": "Gaussian mixture (unsup.)"}

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "axes.axisbelow": True,
})


NAMES = {'fig10_model_comparison': 'fig07_algorithm_comparison', 'fig11_real_maps_by_method': 'figS5_real_maps_nine_methods', 'fig12_leakage_demo': 'fig08_leakage', 'fig14_ggmplus_vs_psg': 'fig03_ggmplus_vs_terrestrial', 'fig16_real_maps_psg_vs_ggm': 'figS6_real_maps_ggmplus_density'}
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


def comparison_table():
    rows = [json.load(open(f)) for f in glob.glob("results/cmp_*.json")]
    t = pd.DataFrame(rows).set_index("model")
    t = t.loc[[m for m in ORDER if m in t.index]]
    t.to_csv("results/model_comparison.csv")
    return t


def fig_comparison(t):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    names = [SHORT[m] for m in t.index]
    y = np.arange(len(t))[::-1]
    ci = pd.read_csv("results/model_comparison_ci.csv").set_index("model").loc[t.index] \
        if os.path.exists("results/model_comparison_ci.csv") else None
    ax = axes[0]
    ax.barh(y, t["A_AP"], height=0.55, color=CAT[0])
    if ci is not None:
        ax.errorbar(t["A_AP"], y, xerr=[t["A_AP"] - ci["AP_lo"], ci["AP_hi"] - t["A_AP"]], fmt="none",
                    ecolor=INK, elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=4)
    ax.axvline(0.055, color=INK2, lw=0.7)
    ax.text(0.057, y[-1] - 0.9, "prevalence", fontsize=6, color=INK2)
    for yy, v, a, hi in zip(y, t["A_AP"], t["A_AUC"], ci["AP_hi"] if ci is not None else t["A_AP"]):
        ax.text(hi + 0.005, yy, f"{v:.3f}  (AUC {a:.2f})", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.33)
    ax.set_xlabel("Average precision, prior A test (95% CI)")
    ax.set_title("a  Cell level", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    if ci is not None:
        ok = t["A_recall@1.0FP"] > 0
        ax.errorbar(t["A_recall@1.0FP"][ok], y[ok.to_numpy()],
                    xerr=[np.clip(t["A_recall@1.0FP"][ok] - ci["recall1_lo"][ok], 0, None),
                          np.clip(ci["recall1_hi"][ok] - t["A_recall@1.0FP"][ok], 0, None)],
                    fmt="none", ecolor=CAT[0], elinewidth=0.8, capsize=1.5, capthick=0.7, zorder=2)
    for c, key, lab in zip(CAT, ["A", "B", "C"], ["prior A (training)", "prior B", "prior C"]):
        ax.scatter(t[f"{key}_recall@1.0FP"], y, s=22, color=c, edgecolors="white", linewidths=0.6, zorder=3,
                   label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("Recall at ≤ 1 false peak / scene")
    ax.set_title("b  Object level & prior transfer", loc="left", color=INK)
    ax.legend(loc="upper left", fontsize=6.3, handletextpad=0.1, borderaxespad=0.3)
    for yy, m in zip(y, t.index):
        if t.loc[m, "A_recall@1.0FP"] == 0:
            ax.text(0.03, yy, "0 (plateau-valued map)", va="center", fontsize=6, color=INK2)
    ax.grid(axis="y", visible=False)
    ax = axes[2]
    ax.barh(y, t["global_p_value"], height=0.55, color=CAT[0])
    ax.axvline(0.05, color=INK2, lw=0.7)
    ax.text(0.052, y[0] + 0.35, "0.05", fontsize=6, color=INK2)
    for yy, v, mp in zip(y, t["global_p_value"], t["real_max_p"]):
        ax.text(v + 0.01, yy, f"{v:.2f}  (max p {mp:.2f})", va="center", fontsize=6, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Global p-value, real terrestrial map")
    ax.set_title("c  Real data", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    for ax in axes:
        ax.set_ylim(-0.7, len(t) - 0.3)                 # same rows in all three panels
    fig.tight_layout()
    save(fig, "fig10_model_comparison")


def fig_real_maps(t):
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    models = list(t.index)
    nr = int(np.ceil(len(models) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(7.2, 2.35 * nr + 0.6), layout="constrained")
    lv = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    cmap = ListedColormap(SEQ)
    ny, nx = fe.shape
    h = np.vstack([fe.hull, fe.hull[:1]]) / 1e3
    for k, (ax, m) in enumerate(zip(axes.ravel(), models)):
        p = np.load(f"results/real_{m}.npy")
        Z = np.full(ny * nx, np.nan)
        Z[fe.inside] = p
        cf = ax.pcolormesh(fe.xs / 1e3, fe.ys / 1e3, Z.reshape(ny, nx), cmap=cmap, norm=BoundaryNorm(lv, cmap.N),
                           shading="nearest")
        ax.plot(h[:, 0], h[:, 1], color=INK2, lw=0.5)
        ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, s=3, c=INK, lw=0, zorder=4)
        for q, lab in [(P1, "P1"), (P2, "P2")]:
            ax.scatter(*q, marker="^", s=18, facecolors="none", edgecolors=INK, linewidths=0.7, zorder=5)
        ax.scatter(*GAJAH, marker="*", s=30, c=CAT[1], edgecolors="white", linewidths=0.4, zorder=6)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(390, 411)
        ax.set_ylim(9129.5, 9151)
        ax.set_xticks([390, 400, 410])
        ax.set_yticks([9130, 9140, 9150])
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.set_title(f"{VS.tok(k)}{SHORT[m]} · p-global {t.loc[m, 'global_p_value']:.2f}", loc="left", color=INK,
                     fontsize=7)
    for ax in axes.ravel()[len(models):]:
        ax.set_visible(False)
    cb = fig.colorbar(cf, ax=list(axes.ravel()), orientation="horizontal", fraction=0.03, pad=0.02, aspect=50,
                      shrink=0.6)
    cb.set_label("Probability of a complex-scale centre zone (GMM: posterior of anomaly component)", color=INK2,
                 fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "fig11_real_maps_by_method")


def fig_leakage():
    s = json.load(open("results/leakage_summary.json"))
    sm = s["single_map_median"]
    pooled = s["pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    groups = [("Accuracy", "accuracy"), ("Balanced acc.", "balanced_acc"), ("ROC-AUC", "AUC")]
    x = np.arange(len(groups))
    w = 0.36
    ax = axes[0]
    a = [sm[f"same_map_test_{k}"] for _, k in groups]
    b = [sm[f"new_maps_{k}"] for _, k in groups]
    ax.bar(x - w / 2, a, w, color=CAT[1], label="random 10% of the same map")
    ax.bar(x + w / 2, b, w, color=CAT[0], label="500 new maps")
    for xx, v in zip(list(x - w / 2) + list(x + w / 2), a + b):
        ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.3, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("a  Single-map 80/10/10 protocol (median of 30 maps)", loc="left", color=INK)
    ax.legend(loc="upper center", ncol=2, fontsize=6.3)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    keys = list(pooled)
    for i, (k, c) in enumerate(zip(keys, [CAT[1], CAT[0]])):
        vals = [pooled[k][f"held_out_cells_{m}"] for _, m in groups]
        ax.bar(x + (i - 0.5) * w, vals, w, color=c, label=k)
        for xx, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.3, color=INK2)
    nl, nh = pooled[keys[0]]["new_maps_AUC"], pooled[keys[1]]["new_maps_AUC"]
    note = f"Same models on 500 new maps: AUC {nl:.2f} (random split) vs {nh:.2f} (scene-wise)"
    if VS.VOLC:   # Volcanica version: below the axis, so that the note does not overprint the bars
        ax.text(0.5, -0.2, note, transform=ax.transAxes, fontsize=6.2, color=INK2, ha="center", va="top")
    else:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=6.2, color=INK2, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("b  Held-out cells: random vs scene-wise split (150 maps)", loc="left", color=INK)
    ax.legend(loc="upper center", ncol=2, fontsize=6.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig12_leakage_demo")


def fig_ggm_value():
    """Value of GGMplus features under naive vs calibrated emulation (dot plot)."""
    g = pd.read_csv("results/ggm_value.csv")
    c = pd.read_csv("results/ggm_value_calibrated.csv")
    ci = pd.read_csv("results/model_comparison_ci.csv").set_index("model").loc["HGB"]
    base = g[g.model == "PSG only"].iloc[0]
    rows = []
    for e in (0.0, 2.0, 5.0, 10.0):
        s = g[g.ggm_err_mGal == e]
        rows.append((f"LP 10 km, error {e:g} mGal", s[s.model == "PSG + GGM"].iloc[0], s[s.model != "PSG + GGM"].iloc[0]))
    n_cal = 0
    for v, lab in [("EGM2008-limit_perfect", "LP 18.3 km (EGM2008 limit), no error"),
                   ("calibrated_most_optimistic", "Calibrated: most optimistic"),
                   ("calibrated_best_fit", "Calibrated: best fit"),
                   ("calibrated_best_fit_short", "Calibrated: best fit + 2 mGal short-scale")]:
        s = c[c.variant == v]
        if len(s):
            rows.append((lab, s[s.model == "PSG + GGM"].iloc[0], s[s.model != "PSG + GGM"].iloc[0]))
            n_cal += v.startswith("calibrated")
    y = np.arange(len(rows))[::-1].astype(float)
    y[-n_cal:] -= 0.5                                 # gap before the calibrated (accepted) variants
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for ax, col, lab, lo, hi in [(axes[0], "recall@1.0FP", "Recall at ≤ 1 false peak / scene", "recall1_lo", "recall1_hi"),
                                 (axes[1], "AP", "Average precision (cell level)", "AP_lo", "AP_hi")]:
        ax.axvspan(ci[lo], ci[hi], color=BAND, zorder=0)
        ax.axvline(base[col], color=INK, lw=0.9, zorder=1)
        for yy, (name, a, b) in zip(y, rows):
            ax.plot([a[col], b[col]], [yy, yy], color=GRID, lw=1.0, zorder=1)
            ax.scatter(a[col], yy, s=26, color=CAT[0], edgecolors="white", linewidths=0.6, zorder=3,
                       label="PSG + GGM" if yy == y[0] else None)
            ax.scatter(b[col], yy, s=26, color=CAT[1], edgecolors="white", linewidths=0.6, zorder=3,
                       label="GGM only (no stations)" if yy == y[0] else None)
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[1].set_yticklabels([])
    yd = y[-n_cal - 1] - 0.75
    axes[0].axhline(yd, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    axes[1].axhline(yd, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    axes[0].text(0.01, yd + 0.1, "↑ rejected by the misfit check", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="bottom", fontsize=6, color=INK2)
    axes[0].text(0.01, yd - 0.1, "↓ consistent with the real misfit", transform=axes[0].get_yaxis_transform(),
                 ha="left", va="top", fontsize=6, color=INK2)
    axes[0].text(base["recall@1.0FP"], y[0] + 0.75, "PSG only (95% CI)", fontsize=6, color=INK2, ha="center")
    axes[0].set_ylim(y[-1] - 0.8, y[0] + 1.2)
    axes[1].set_ylim(y[-1] - 0.8, y[0] + 1.2)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.3, handletextpad=0.2)
    axes[0].set_title("a  Object level", loc="right", color=INK)
    axes[1].set_title("b  Cell level", loc="right", color=INK)
    fig.tight_layout()
    save(fig, "fig13_ggm_value")


def fig_ggm_calibration():
    """Is the emulated GGMplus consistent with the real PSG-GGMplus misfit?"""
    import ggm_emulation as ge
    cal = pd.read_csv("results/ggm_emulation_calibration2.csv")
    real = json.load(open("results/ggm_real_stats_for_calibration.json"))
    xy, hull, fe, gg = ge.setup()
    z = np.load("results/F_A_test.npz")
    F, D = z["F"].astype(float), z["D"]
    key = ge.KEYS["A_test"][0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax = axes[0]
    for (lam, a, sl, ss, lab), col in zip([(10_000.0, 1.0, 5.0, 0.0, "LP 10 km, a = 1, 5 mGal"),
                                           (18_300.0, 1.0, 0.0, 0.0, "LP 18.3 km, a = 1, no error"),
                                           (45_000.0, 0.33, 10.0, 0.0, "LP 45 km, a = 0.33, 10 mGal (best fit)")],
                                          CAT):
        G = ge.emulate2(gg, F, key, 0, lam, a, sl, ss)
        st = ge.misfit_stats(gg, xy, D, G)
        ax.scatter(st["trend_sd_ratio"], st["resid_r"], s=4, color=col, alpha=0.35, lw=0, label=lab)
    ax.scatter(real["H1_psg_2670"]["trend_sd_ratio"], real["H1_psg_2670"]["resid_r"], marker="*", s=110, c=INK,
               edgecolors="white", linewidths=0.6, zorder=5, label="real (PSG as is)")
    ax.scatter(real["H2_psg_2840_converted"]["trend_sd_ratio"], real["H2_psg_2840_converted"]["resid_r"], marker="*",
               s=90, facecolors="white", edgecolors=INK, linewidths=0.8, zorder=5, label="real (PSG 2.84 → 2.67)")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlabel("GGM / PSG amplitude of the 2nd-order trend")
    ax.set_ylabel("Correlation of residuals after trend")
    ax.set_title("a  Synthetic scenes vs real misfit", loc="left", color=INK)
    from matplotlib.lines import Line2D
    labs = ["LP 10 km, a = 1, 5 mGal", "LP 18.3 km, a = 1, no error", "LP 45 km, a = 0.33, 10 mGal (best fit)"]
    hs = [Line2D([], [], ls="", marker="o", ms=4, mfc=c, mec="none") for c in CAT]
    hs += [Line2D([], [], ls="", marker="*", ms=8, mfc=INK, mec="white"),
           Line2D([], [], ls="", marker="*", ms=7.5, mfc="white", mec=INK)]
    ax.legend(hs, labs + ["real, PSG as is (H1)", "real, PSG 2.84 → 2.67 (H2)"], loc="lower right", fontsize=5.8,
              handletextpad=0.2, borderaxespad=0.2)
    ax = axes[1]
    lams, As = [10.0, 18.3, 30.0, 45.0], [1.0, 0.5, 0.33]
    M = np.array([[cal[(cal.lambda_km == l) & (cal.a == a)].min_tail_pct_H1.max() for a in As] for l in lams])
    lv = [0, 2.5, 5, 10, 20, 50]
    cmap = ListedColormap(SEQ[:5])
    ax.pcolormesh(np.arange(4) - 0.5, np.arange(5) - 0.5, M, cmap=cmap, norm=BoundaryNorm(lv, cmap.N))
    for i in range(len(lams)):
        for j in range(len(As)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] >= 10 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{a:g}" for a in As])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{l:g} km" for l in lams])
    ax.set_xlabel("Amplitude factor a")
    ax.set_ylabel("Low-pass cut-off wavelength λ (half amplitude)")
    ax.grid(False)
    ax.set_title("b  Typicality of the real misfit (%)", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig15_ggm_calibration")


def fig_real_ggm_maps():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    z = np.load("results/real_ggm_maps.npz")
    s = json.load(open("results/real_ggm_summary.json"))
    v = "calibrated_best_fit_short"
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.6), layout="constrained")
    lv = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    cmap = ListedColormap(SEQ)
    ny, nx = fe.shape
    h = np.vstack([fe.hull, fe.hull[:1]]) / 1e3
    for i, (hyp, hlab) in enumerate([("H1", "CBA as supplied (H1)"), ("H2", VS.si("CBA 2.84 → 2.67 g/cm³ (H2)"))]):
        for j, (k, sk, title) in enumerate([(f"PSG_only_{hyp}", f"PSG only | {hyp}", "Terrestrial only"),
                                            (f"PSG_GGM_{v}_{hyp}", f"PSG + GGM ({v}) | {hyp}", "Terrestrial + GGMplus")]):
            ax = axes[i, j]
            Z = np.full(ny * nx, np.nan)
            Z[fe.inside] = z[k]
            cf = ax.pcolormesh(fe.xs / 1e3, fe.ys / 1e3, Z.reshape(ny, nx), cmap=cmap,
                               norm=BoundaryNorm(lv, cmap.N), shading="nearest")
            ax.plot(h[:, 0], h[:, 1], color=INK2, lw=0.5)
            ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, s=3, c=INK, lw=0, zorder=4)
            for q in (P1, P2):
                ax.scatter(*q, marker="^", s=18, facecolors="none", edgecolors=INK, linewidths=0.7, zorder=5)
            ax.scatter(*GAJAH, marker="*", s=30, c=CAT[1], edgecolors="white", linewidths=0.4, zorder=6)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_xlim(390, 411)
            ax.set_ylim(9129.5, 9151)
            ax.set_xticks([390, 400, 410])
            ax.set_yticks([9130, 9140, 9150])
            ax.ticklabel_format(useOffset=False, style="plain")
            extra = f" · r = {s[sk]['corr_with_psg_only_map']:.2f}" if j == 1 else ""
            ax.set_title(f"{'abcd'[2 * i + j]}  {title} · {hlab}\nmax p {s[sk]['max_p']:.2f} · "
                         f"p-global {s[sk]['global_p']:.2f}{extra}", loc="left", color=INK, fontsize=7)
    cb = fig.colorbar(cf, ax=list(axes.ravel()), orientation="horizontal", fraction=0.04, pad=0.02, aspect=40,
                      shrink=0.7)
    cb.set_label("Probability of a complex-scale centre zone", color=INK2, fontsize=7)
    cb.outline.set_visible(False)
    save(fig, "fig16_real_maps_psg_vs_ggm")


def fig_ggm_rq1():
    """GGMplus re-reduced CBA vs PSG CBA."""
    import ggm_real as gr
    z = np.load("results/ggm_real_reduced.npz")
    res = json.load(open("results/ggm_real_comparison.json"))["comparison"]
    xy, df = load_psg("data/gravity_requested_area.csv")
    cba = df.CBA.to_numpy()
    ny, nx = [int(v) for v in z["shape"]]
    X, Y, C = (z[k].reshape(ny, nx) / (1e3 if k in ("x", "y") else 1) for k in ("x", "y", "cba"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
    ax = axes[0]
    lv = np.arange(80, 161, 10)
    cmap = ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6"])
    cf = ax.pcolormesh(X, Y, C, cmap=cmap, norm=BoundaryNorm(lv, cmap.N), shading="nearest")
    ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, c=cba, cmap=cmap, norm=BoundaryNorm(lv, cmap.N), s=24,
               edgecolors=INK, linewidths=0.5, zorder=4)
    h = convex_hull(xy) / 1e3
    h = np.vstack([h, h[:1]])
    ax.plot(h[:, 0], h[:, 1], color=INK, lw=0.6)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xticks([380, 390, 400, 410, 420])
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("a  GGMplus CBA (re-reduced) · dots: terrestrial", loc="left", color=INK)
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("CBA (mGal)", color=INK2, fontsize=7)
    cb.outline.set_visible(False)
    v = gr.at_points(z["cba"], z["x"], z["y"], xy[:, 0], xy[:, 1])
    ax = axes[1]
    ax.plot([90, 150], [90, 150], color=AXIS, lw=0.8)
    ax.scatter(v, cba, s=14, color=CAT[0], edgecolors="white", linewidths=0.5)
    r = res["GGM CBA (all wavelengths)"]
    ax.text(0.03, 0.97, f"r = {r['r']:.2f}\nbias = {r['bias']:.1f} mGal\nSD diff = {r['sd_diff']:.1f} mGal",
            transform=ax.transAxes, va="top", fontsize=6.3, color=INK2)
    ax.set_xlabel("GGMplus CBA at station (mGal)")
    ax.set_ylabel("Terrestrial CBA (mGal)")
    ax.set_title("b  All wavelengths", loc="left", color=INK)
    ax = axes[2]
    tp, _ = gr.poly2(xy, cba)
    tg, _ = gr.poly2(xy, v)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.scatter(v - tg, cba - tp, s=14, color=CAT[0], edgecolors="white", linewidths=0.5)
    rr = res["residual after trend"]
    ax.text(0.03, 0.97, f"r = {rr['r']:.2f}\nSD terrestrial = {rr['psg_sd']:.1f}\nSD GGMplus = {rr['ggm_sd']:.1f} mGal",
            transform=ax.transAxes, va="top", fontsize=6.3, color=INK2)
    ax.set_xlabel("GGMplus residual (mGal)")
    ax.set_ylabel("Terrestrial residual (mGal)")
    ax.set_title("c  After 2nd-order trend", loc="left", color=INK)
    fig.tight_layout()
    save(fig, "fig14_ggmplus_vs_psg")


def fig_ggmplus_value_composite():
    """Fig. 9: (a, b) calibration of the GGMplus emulation against the real misfit; (c, d) value of
    GGMplus-derived features under naive and calibrated emulations."""
    import ggm_emulation as ge
    from matplotlib.lines import Line2D
    cal = pd.read_csv("results/ggm_emulation_calibration2.csv")
    real = json.load(open("results/ggm_real_stats_for_calibration.json"))
    xy, hull, fe, gg = ge.setup()
    z = np.load("results/F_A_test.npz")
    F, D = z["F"].astype(float), z["D"]
    key = ge.KEYS["A_test"][0]
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 1.02], hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    labs = ["LP 10 km, a = 1, 5 mGal", "LP 18.3 km, a = 1, no error", "LP 45 km, a = 0.33, 10 mGal (best fit)"]
    for (lam, a, sl, ss), col in zip([(10_000.0, 1.0, 5.0, 0.0), (18_300.0, 1.0, 0.0, 0.0), (45_000.0, 0.33, 10.0, 0.0)], CAT):
        st = ge.misfit_stats(gg, xy, D, ge.emulate2(gg, F, key, 0, lam, a, sl, ss))
        ax.scatter(st["trend_sd_ratio"], st["resid_r"], s=3.5, color=col, alpha=0.35, lw=0)
    r1, r2 = real["H1_psg_2670"], real["H2_psg_2840_converted"]
    ax.scatter(r1["trend_sd_ratio"], r1["resid_r"], marker="*", s=110, c=INK, edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter(r2["trend_sd_ratio"], r2["resid_r"], marker="*", s=90, facecolors="white", edgecolors=INK, linewidths=0.8, zorder=5)
    hs = [Line2D([], [], ls="", marker="o", ms=4, mfc=c, mec="none") for c in CAT]
    hs += [Line2D([], [], ls="", marker="*", ms=8, mfc=INK, mec="white"), Line2D([], [], ls="", marker="*", ms=7.5, mfc="white", mec=INK)]
    ax.legend(hs, labs + ["real data, H1 (CBA as supplied)", VS.si("real data, H2 (2.84 → 2.67 g/cm³)")], loc="lower right",
              fontsize=5.6, handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.6, 1.0)
    ax.set_xlabel("GGMplus / terrestrial amplitude of 2nd-order trend")
    ax.set_ylabel("Correlation of residuals after trend")
    ax.set_title("a  Synthetic scenes vs real misfit", loc="left", color=INK)
    ax = fig.add_subplot(gs[0, 1])
    lams, As = [10.0, 18.3, 30.0, 45.0], [1.0, 0.5, 0.33]
    M = np.array([[cal[(cal.lambda_km == l) & (cal.a == a)].min_tail_pct_H1.max() for a in As] for l in lams])
    lv = [0, 2.5, 5, 10, 20, 50]
    cmap = ListedColormap(SEQ[:5])
    ax.pcolormesh(np.arange(4) - 0.5, np.arange(5) - 0.5, M, cmap=cmap, norm=BoundaryNorm(lv, cmap.N))
    for i in range(len(lams)):
        for j in range(len(As)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] >= 10 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{a:g}" for a in As])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{l:g} km" for l in lams])
    ax.set_xlabel("Amplitude factor a")
    ax.set_ylabel("Low-pass cut-off wavelength λ (half amplitude)")
    ax.grid(False)
    ax.set_title("b  Typicality of the real misfit (%)", loc="left", color=INK)
    # --- value of information
    g = pd.read_csv("results/ggm_value.csv")
    c = pd.read_csv("results/ggm_value_calibrated.csv")
    ci = pd.read_csv("results/model_comparison_ci.csv").set_index("model").loc["HGB"]
    base = g[g.model == "PSG only"].iloc[0]
    rows = []
    for e in (0.0, 2.0, 5.0, 10.0):
        s_ = g[g.ggm_err_mGal == e]
        rows.append((f"LP 10 km, {e:g} mGal", s_[s_.model == "PSG + GGM"].iloc[0], s_[s_.model != "PSG + GGM"].iloc[0]))
    for v, lab in [("EGM2008-limit_perfect", "LP 18.3 km, no error"), ("calibrated_most_optimistic", "Calibrated, most optimistic"),
                   ("calibrated_best_fit", "Calibrated, best fit"), ("calibrated_best_fit_short", "Calibrated, best fit + 2 mGal short")]:
        s_ = c[c.variant == v]
        rows.append((lab, s_[s_.model == "PSG + GGM"].iloc[0], s_[s_.model != "PSG + GGM"].iloc[0]))
    n_cal = 3
    y = np.arange(len(rows))[::-1].astype(float)
    y[-n_cal:] -= 0.5
    yd = y[-n_cal - 1] - 0.75
    axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, col, lab, lo, hi in [(axes[0], "recall@1.0FP", "Recall at ≤ 1 false peak / scene", "recall1_lo", "recall1_hi"),
                                 (axes[1], "AP", "Average precision", "AP_lo", "AP_hi")]:
        ax.axvspan(ci[lo], ci[hi], color=BAND, zorder=0)
        ax.axvline(base[col], color=INK, lw=0.9, zorder=1)
        for yy, (name, a, b) in zip(y, rows):
            ax.plot([a[col], b[col]], [yy, yy], color=GRID, lw=1.0, zorder=1)
            ax.scatter(a[col], yy, s=24, color=CAT[0], edgecolors="white", linewidths=0.6, zorder=3,
                       label="Terrestrial + GGMplus" if yy == y[0] else None)
            ax.scatter(b[col], yy, s=24, color=CAT[1], edgecolors="white", linewidths=0.6, zorder=3,
                       label="GGMplus only (no stations)" if yy == y[0] else None)
        ax.axhline(yd, color=AXIS, lw=0.6, ls=(0, (3, 2)))
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
        ax.set_ylim(y[-1] - 0.8, y[0] + 1.3)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=6.5)
    axes[1].set_yticklabels([])
    axes[0].text(0.01, yd + 0.1, "↑ rejected by misfit check", transform=axes[0].get_yaxis_transform(), ha="left", va="bottom",
                 fontsize=5.8, color=INK2)
    axes[0].text(0.01, yd - 0.1, "↓ consistent with real misfit", transform=axes[0].get_yaxis_transform(), ha="left", va="top",
                 fontsize=5.8, color=INK2)
    axes[0].text(base["recall@1.0FP"], y[0] + 0.8, "terrestrial only (95% CI)", fontsize=5.8, color=INK2, ha="center",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=6)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=2, fontsize=6.2, handletextpad=0.2, columnspacing=1.0)
    axes[0].set_title("c  Object level", loc="right", color=INK)
    axes[1].set_title("d  Cell level", loc="right", color=INK)
    save(fig, "fig09_ggmplus_value")


if __name__ == "__main__":
    t = comparison_table()
    fig_comparison(t)
    fig_real_maps(t)
    fig_leakage()
    fig_ggm_rq1()
    fig_real_ggm_maps()
    fig_ggmplus_value_composite()
    print("figures done")
