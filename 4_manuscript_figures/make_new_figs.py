"""Fig. 1 (study area and data) and Fig. 2 (workflow) of the manuscript."""
from __future__ import annotations

import logging
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
sys.path.insert(0, "../3_comparison_ggmplus")
sys.path.insert(0, "..")  # repository root: volcanica_figstyle.py
from geometry import convex_hull  # noqa: E402
from ggm_real import load_grid  # noqa: E402

import os
OUT = os.environ.get("FIG_OUT", "../figures_manuscript")
os.makedirs(OUT, exist_ok=True)
import volcanica_figstyle as VS  # noqa: E402  (PANEL_STYLE=volcanica: upper-case 12 pt panel letters)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SEQ8 = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6"]
plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300,
})
GAJAH = (401.448, 9140.279)
MENOREH = [(404.150, 9155.957), (410.109, 9154.641), (406.139, 9154.192)]      # highest grid nodes (954, 947, 912 m)


def save(fig, name):
    out = OUT
    if VS.VOLC:   # Volcanica template: upper-case 12 pt panel letters, written to the Volcanica figure folder
        VS.relabel(fig)
        out = VS.out_dir(OUT)
    fig.savefig(f"{out}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_study_area():
    import geopandas as gpd
    from matplotlib.colors import LinearSegmentedColormap
    from pyproj import Transformer
    x, y, z, _ = load_grid("../3_comparison_ggmplus/data/ggmplus_oasis.xls")
    st = pd.read_csv("../3_comparison_ggmplus/data/gravity_requested_area.csv")
    xy = st[["X", "Y"]].to_numpy()
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    X, Y = x / 1e3, y / 1e3
    ls = LightSource(azdeg=315, altdeg=40)
    dx = np.median(np.diff(x, axis=1))
    hs = ls.hillshade(z, vert_exag=1.5, dx=dx, dy=dx)
    ax.pcolormesh(X, Y, np.where(z <= 0, np.nan, 1.0), cmap=ListedColormap(["#f3f2ec"]), shading="nearest", zorder=0)
    ax.pcolormesh(X, Y, np.ma.masked_where(z <= 0, hs), cmap="Greys_r", vmin=0.0, vmax=1.35, shading="nearest",
                  alpha=0.55, zorder=1)
    ax.contour(X, Y, z, levels=[250, 500, 750], colors=[MUTED], linewidths=0.35, zorder=2)
    hull = convex_hull(xy) / 1e3
    hull = np.vstack([hull, hull[:1]])
    ax.plot(hull[:, 0], hull[:, 1], color=INK, lw=0.8, zorder=3)
    ax.add_patch(Rectangle((390, 9130), 20, 20, fill=False, ec=INK2, lw=0.7, ls=(0, (4, 2)), zorder=3))
    cmap = LinearSegmentedColormap.from_list("seq", SEQ8)
    sc = ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, c=st.CBA, cmap=cmap, vmin=100, vmax=150, s=24,
                    edgecolors=INK, linewidths=0.5, zorder=5)
    ax.scatter(*GAJAH, marker="*", s=95, c=CAT[1], edgecolors="white", linewidths=0.6, zorder=6)
    ax.annotate("Gajah summit", GAJAH, xytext=(0, 9), textcoords="offset points", ha="center", fontsize=6.8,
                color=INK, zorder=7)
    for q in MENOREH:
        ax.scatter(*q, marker="^", s=24, facecolors="white", edgecolors=INK, linewidths=0.7, zorder=6)
    ax.text(406.4, 9157.0, "Menoreh highlands (912–954 m)", ha="center", fontsize=6.8, color=INK, zorder=7)
    ax.text(390.3, 9150.5, "data window (390–410 km E, 9130–9150 km N)", fontsize=6.2, color=INK2, zorder=7)
    ax.text(410.8, 9133.0, "convex hull of\nthe 43 stations\n(345.6 km²)", fontsize=6.2, color=INK, zorder=7)
    ax.text(381.0, 9128.3, "Indian Ocean", fontsize=6.8, color=MUTED, style="italic", zorder=7)
    ax.plot([413, 418], [9128.8, 9128.8], color=INK, lw=1.6, solid_capstyle="butt", zorder=7)
    ax.text(415.5, 9129.4, "5 km", ha="center", fontsize=6.5, color=INK, zorder=7)
    ax.annotate("", xy=(420.6, 9169.3), xytext=(420.6, 9166.3), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.text(420.6, 9165.2, "N", ha="center", va="top", fontsize=7, color=INK)
    ax.set_xlim(x.min() / 1e3, x.max() / 1e3)
    ax.set_ylim(y.min() / 1e3, y.max() / 1e3)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Easting (km, UTM zone 49S)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, ticks=[100, 110, 120, 130, 140, 150])
    cb.set_label("Terrestrial complete Bouguer anomaly (mGal)", color=INK2, fontsize=7)
    cb.outline.set_visible(False)
    # location inset (Natural Earth 1:110m)
    ins = ax.inset_axes([0.015, 0.735, 0.40, 0.25])
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.plot(ax=ins, color="#e9e8e2", edgecolor=MUTED, linewidth=0.4)
    tr = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([x.min(), x.max(), x.max(), x.min(), x.min()], [y.min(), y.min(), y.max(), y.max(), y.min()])
    ins.fill(lon, lat, facecolor=CAT[1], edgecolor=CAT[1], lw=1.0, zorder=4)
    ins.text(108.0, -7.05, "Java", fontsize=6.3, color=INK2, ha="center", style="italic")
    ins.set_xlim(105.0, 115.0)
    ins.set_ylim(-9.0, -5.6)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.text(0.98, 0.06, "105–115°E, 5.6–9°S", transform=ins.transAxes, ha="right", fontsize=5.3, color=MUTED)
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(AXIS)
    ins.grid(False)
    save(fig, "fig01_study_area")


def box(ax, x, y, w, h, text, tag, fc=BAND, ec=AXIS, fs=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2 - 0.016, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.22)
    ax.text(x + 0.008, y + h - 0.01, tag, ha="left", va="top", fontsize=6.3, color=INK2, fontweight="bold")


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.7, shrinkA=1, shrinkB=1))


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.06, 1.0)
    ax.axis("off")
    W, H = 0.31, 0.15
    X = [0.01, 0.345, 0.68]
    heads = ["1  Data and GGMplus reliability (RQ1)", "2  Simulation-trained detection (RQ2)",
             "3  GGMplus value and application (RQ2, RQ3)"]
    rows = [0.795, 0.615, 0.435, 0.255, 0.075]
    cols = [["Terrestrial CBA\n43 stations, median spacing 2.4 km",
             "GGMplus grid (220 m)\ngravity and surface height",
             "Re-reduction of GGMplus\nfree-air anomaly − prism topographic\neffect (2.67 g/cm³, full tile)",
             "Band-wise comparison\nall λ; λ > 10 km; λ > 18.3 km;\n2nd-order trend and residual",
             "Reduction-density hypotheses\nH1: CBA as supplied\nH2: 2.84 → 2.67 g/cm³"],
            ["Scene generator at the real geometry\ntargets R = 1.5–4 km, distractors,\nregional field, noise, outliers",
             "Prior predictive check\npriors A, B, C vs real residual statistics",
             "26 coordinate-free features per cell\nmap derivatives, station rings,\nannulus contrast, matched filters",
             "Nine learners, scene-wise splits\n4000 train / 800 validation / 1000 test",
             "Cell (AUC, AP, ECE) and object (FROC)\nmetrics, paired scene bootstrap,\nleakage experiment"],
            ["GGMplus emulation\nG = a · LP_λ(true field) + errors\n(144 variants)",
             "Misfit check against 1d\n11 statistics of the real\nterrestrial–GGMplus comparison",
             "Value of GGMplus features\npaired differences and\n'GGMplus only' leakage diagnostic",
             "Real data: probability maps\nglobal test vs 1000 target-free scenes",
             "Uncertainty: ensemble, noise,\nstation jackknife, prior, density (1e);\ncovariate-shift check"]]
    for j, (x0, head, texts) in enumerate(zip(X, heads, cols)):
        ax.text(x0, 0.985, head, fontsize=7.2, fontweight="bold", color=INK, va="top")
        for i, (yy, t) in enumerate(zip(rows, texts)):
            box(ax, x0, yy, W, H, t, f"{j + 1}{'abcde'[i]}")
            if i:
                arrow(ax, (x0 + W / 2, rows[i - 1]), (x0 + W / 2, yy + H))
    arrow(ax, (X[0] + W, rows[0] + H / 2), (X[1], rows[0] + H / 2))
    arrow(ax, (X[1] + W, rows[0] + H / 2), (X[2], rows[0] + H / 2))
    arrow(ax, (X[1] + W, rows[3] + H / 2), (X[2], rows[3] + H / 2))
    save(fig, "fig02_workflow")


if __name__ == "__main__":
    fig_study_area()
    fig_workflow()
    print("done")
