"""Figures for the synthetic detectability test (PNG 300 dpi + vector PDF)."""
from __future__ import annotations

import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
from cylinder import Profile  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, "..")  # repository root: volcanica_figstyle.py
import volcanica_figstyle as VS  # noqa: E402  (PANEL_STYLE=volcanica: upper-case 12 pt panel letters)
from geometry import convex_hull, dist_to_boundary, grid_nodes, inside, load_psg, pairwise_dist  # noqa: E402

# ---- palette (dataviz reference instance; ordinal ramps validated with --ordinal) ----
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
ACCENT = "#eb6834"                                                   # categorical slot 2 (PSG-43)
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]      # ordinal, 5 steps (PASS)
RAMP3 = ["#86b6ef", "#2a78d6", "#104281"]                            # ordinal, 3 steps (PASS)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # sequential classes

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})

meta = json.load(open("results/meta.json"))
df = pd.read_csv("results/pod_results.csv")
H = meta["H"]
d_med = meta["context"]["body_to_nearest_station_m"]["median"] / 1e3
d_p90 = meta["context"]["body_to_nearest_station_m"]["p90"] / 1e3
thr2_psg = pd.read_csv("results/null_thresholds.csv").query("geometry == 'PSG-43'").thr_D2_sigma.iloc[0]


NAMES = {'fig1_anomaly_profiles': 'figS1_anomaly_profiles', 'fig3_pod_curves': 'figS2_pod_curves', 'fig4_R80_vs_spacing': 'fig04_detectability_R80', 'fig2_station_coverage': 'figX_station_coverage'}
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


# ---------------- Fig 1: anomaly profiles vs typical station distance ----------------
def fig_profiles():
    fig, ax = plt.subplots(figsize=(3.54, 2.7))
    r = np.linspace(0, 6000, 601)
    radii = [250, 500, 1000, 2000, 3000]
    ax.axvspan(d_med, d_p90, color=BAND, lw=0, zorder=0)
    ax.text(0.08, 21.3, f"shaded: distance to nearest terrestrial station (median {d_med:.1f}–P90 {d_p90:.1f} km)",
            ha="left", va="top", fontsize=6.3, color=INK2)
    for sigma in (2.0, 5.5):
        y = thr2_psg * sigma
        ax.axhline(y, color=INK2, lw=0.7, zorder=1)
        ax.text(5.95, y + 0.2, f"detection level\nσ = {sigma:g} mGal → {y:.1f} mGal", ha="right", va="bottom",
                fontsize=6.3, color=INK2, linespacing=1.1)
    for R, c in zip(radii, RAMP5):
        g = 300.0 * Profile(R, 100.0, 100.0 + H)(r)
        ax.plot(r / 1e3, g, color=c, lw=1.5, label=f"R = {R / 1e3:g} km", zorder=3)
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 21.5)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Horizontal distance from body axis (km)")
    ax.set_ylabel("Gravity anomaly (mGal)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), handlelength=1.6)
    ax.set_title(VS.si("Vertical cylinder, top 0.1 km, height 2 km, Δρ = +0.3 g/cm³"), loc="left", color=INK)
    save(fig, "fig1_anomaly_profiles")


# ---------------- Fig 2: station coverage map ----------------
def fig_coverage():
    xy, _ = load_psg("data/gravity_requested_area.csv")
    hull = convex_hull(xy)
    nodes, (xs, ys, mask) = grid_nodes(hull, 100.0)
    dist = pairwise_dist(nodes, xy).min(1) / 1e3
    Z = np.full(mask.shape, np.nan)
    Z[mask] = dist
    fig, ax = plt.subplots(figsize=(3.54, 4.1))
    ax.grid(False)
    levels = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5]
    cmap = ListedColormap(SEQ)
    cf = ax.contourf(xs / 1e3, ys / 1e3, Z, levels=levels, cmap=cmap, norm=BoundaryNorm(levels, cmap.N))
    hp = np.vstack([hull, hull[:1]]) / 1e3
    ax.plot(hp[:, 0], hp[:, 1], color=INK2, lw=0.6)
    # body-placement region (>= 2 km inside the hull)
    Zi = np.full(mask.shape, np.nan)
    Zi[mask] = dist_to_boundary(hull, nodes)
    ax.contour(xs / 1e3, ys / 1e3, Zi, levels=[meta["buffer"]], colors=[INK], linewidths=0.6)
    ax.plot([390, 410, 410, 390, 390], [9130, 9130, 9150, 9150, 9130], color=MUTED, lw=0.6,
            label="data window")
    ax.plot([], [], color=INK, lw=0.6, label="Body placement (≥ 2 km inside hull)")
    ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, s=11, c=ACCENT, edgecolors="white", linewidths=0.6, zorder=4,
               label="terrestrial station (n = 43)")
    ax.scatter([401.448], [9140.279], marker="^", s=28, c=INK, edgecolors="white", linewidths=0.6, zorder=5)
    ax.annotate("Gajah summit", (401.448, 9140.279), xytext=(5, 3), textcoords="offset points",
                fontsize=6.5, color=INK, ha="left", va="bottom")
    peaks = np.array([[404.150, 9155.957], [410.109, 9154.641], [406.139, 9154.192]])
    ax.scatter(peaks[:, 0], peaks[:, 1], marker="^", s=24, facecolors="white", edgecolors=INK, linewidths=0.8,
               zorder=5)
    ax.text(411.8, 9153.55, "Menoreh highlands (912–954 m),\noutside the data window", fontsize=6.5, color=INK,
            ha="right", va="top")
    ax.set_aspect("equal")
    ax.set_xlim(388, 412)
    ax.set_ylim(9128, 9161.5)
    ax.set_xlabel("Easting (km, UTM 49S)")
    ax.set_ylabel("Northing (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
    cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.045, pad=0.13, aspect=30)
    cb.set_label("Distance to nearest terrestrial station (km)", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5)
    h, l = ax.get_legend_handles_labels()
    order = [l.index("terrestrial station (n = 43)"), l.index("data window"),
             l.index("Body placement (≥ 2 km inside hull)")]
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="upper left", fontsize=6.5,
              handletextpad=0.4, handlelength=1.4, borderaxespad=0.2, labelspacing=0.3)
    save(fig, "fig2_station_coverage")


# ---------------- Fig 3: POD curves (idealised detector D1) ----------------
def fig_pod(detector="D1", drho=300.0, name="fig3_pod_curves"):
    geos = ["Grid 2.00 km", "Grid 1.41 km", "Grid 1.00 km", "Grid 0.71 km", "Grid 0.50 km"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5), sharex=True, sharey=True)
    for i, sigma in enumerate([2.0, 5.5]):
        for j, zt in enumerate([100, 500, 1000]):
            ax = axes[i, j]
            sub = df[(df.sigma == sigma) & (df.zt_m == zt) & (df.drho == drho)]
            ax.axhline(0.8, color=INK2, lw=0.7, zorder=1)
            for g, c in zip(geos, RAMP5):
                s = sub[sub.geometry == g].sort_values("R_m")
                ax.plot(s.R_m / 1e3, s[f"pod_{detector}"], color=c, lw=1.2, marker="o", ms=2.6,
                        markeredgewidth=0, label=g.replace("Grid ", "grid "), zorder=2)
            s = sub[sub.geometry == "PSG-43"].sort_values("R_m")
            ax.plot(s.R_m / 1e3, s[f"pod_{detector}"], color=ACCENT, lw=1.8, marker="o", ms=3.2,
                    markeredgecolor="white", markeredgewidth=0.6, label="43-station network (actual)", zorder=3)
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
    axes[0, 0].text(0.26, 0.82, "80%", fontsize=6.5, color=INK2, va="bottom")
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [len(h) - 1] + list(range(len(h) - 1))
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.035), handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, name)


# ---------------- Fig 4: design curve R80 vs station spacing ----------------
def fig_r80(detector="D1", drho=300.0, name="fig4_R80_vs_spacing"):
    t = pd.read_csv("results/R80_table.csv")
    t = t[(t.detector == detector) & (t.drho == drho)]
    sp = {"Grid 2.83 km": 2.83, "Grid 2.00 km": 2.0, "Grid 1.41 km": 1.41, "Grid 1.00 km": 1.0,
          "Grid 0.71 km": 0.71, "Grid 0.50 km": 0.5}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for k, (ax, sigma) in enumerate(zip(axes, [2.0, 5.5])):
        ax.axhspan(0, 0.5, color=BAND, lw=0, zorder=0)
        ax.text(0.47, 0.25, "necks & plugs (R ≤ 0.5 km)", fontsize=6.5, color=INK2, va="center")
        for zt, c in zip([100, 500, 1000], RAMP3):
            s = t[(t.sigma == sigma) & (t.zt_m == zt) & (t.geometry.isin(sp))].copy()
            s["x"] = s.geometry.map(sp)
            s = s.sort_values("x")
            y = s.R80_m / 1e3
            cens = s.flag == ">"
            ax.plot(s.x, y, color=c, lw=1.3, zorder=2, label=f"top at {zt / 1e3:g} km")
            ax.scatter(s.x[~cens], y[~cens], s=14, color=c, edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter(s.x[cens], y[cens], s=14, facecolors="white", edgecolors=c, linewidths=0.9, zorder=3)
            p = t[(t.sigma == sigma) & (t.zt_m == zt) & (t.geometry == "PSG-43")].iloc[0]
            dx = {100: 0.09, 500: 0.15, 1000: 0.21}[zt]
            ax.scatter([2.83 + dx], [p.R80_m / 1e3], marker="D", s=16,
                       facecolors=ACCENT if p.flag != ">" else "white", edgecolors=ACCENT, linewidths=0.9,
                       zorder=4)
        ax.set_xlim(0.4, 3.12)
        ax.set_ylim(0, 4.35)
        ax.set_xticks([0.5, 0.71, 1.0, 1.41, 2.0, 2.83])
        ax.set_xticklabels(["0.5", "0.71", "1.0", "1.41", "2.0", "2.83"])
        ax.set_xlabel("Station spacing (km)")
        ax.set_title(f"{VS.tok(k)}σ = {sigma:g} mGal", loc="left", color=INK)
        if (t[(t.sigma == sigma)].flag == ">").any():
            ax.text(2.75, 4.28, "hollow = R80 > 4 km (not reached)", fontsize=6.2, color=INK2, ha="right", va="top")
    axes[0].set_ylabel("R80: radius detected with 80% probability (km)")
    axes[0].scatter([], [], marker="D", s=16, color=ACCENT, label="43-station network (left→right: top 0.1, 0.5, 1 km)")
    axes[0].legend(loc="upper left", handlelength=1.6)
    fig.tight_layout()
    save(fig, name)


if __name__ == "__main__":
    fig_profiles()
    fig_pod()
    fig_r80()
    print("figures written")
