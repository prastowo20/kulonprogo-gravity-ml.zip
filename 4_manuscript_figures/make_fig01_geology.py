"""Figure 1 with a geology panel (revision of 5 Oct 2026, Volcanica manuscript).

(a) Simplified geology of the data window (390-410 km E, 9130-9150 km N) from the digital geological map of
    Indonesia (Pusat Penelitian dan Pengembangan Geologi, Geological Agency of Indonesia; layers Geologi.shp and
    faultline.shp, unit codes as in the source), with the 43 terrestrial stations, their convex hull and the
    Gajah summit.
(b) Study area and terrestrial CBA on the GGMplus surface heights, as in the original Figure 1.
The geological map data are not redistributed: set GEOLOGY_DIR to the folder holding Geologi.shp and faultline.shp.
Output: $FIG_OUT/fig01_study_area_geology.{png,pdf} (default ../figures_manuscript; with PANEL_STYLE=volcanica in
$FIG_OUT/volcanica); a summary of the units is printed (areas inside the hull).
Run in 4_manuscript_figures: GEOLOGY_DIR=/path/to/folder python make_fig01_geology.py
"""
from __future__ import annotations

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pyproj import Transformer
from shapely.geometry import MultiPoint, box

from make_new_figs import (AXIS, CAT, GAJAH, INK, INK2, MENOREH, MUTED, SEQ8, VS, convex_hull, load_grid, save)

GEOLOGY_DIR = os.environ.get("GEOLOGY_DIR", "")   # folder with Geologi.shp and faultline.shp (not redistributed)
WINDOW = (390000, 9130000, 410000, 9150000)
BBOX_LONLAT = (109.94, -7.93, 110.24, -7.63)          # covers the 44 km GGMplus tile used in panel (b)

# unit code -> (legend label, colour); order = legend order (young to old, then igneous units)
UNITS = {
    "Qac": ("Qac  Coastal deposits (Holocene)", "#f7f0c6"),
    "Qa": ("Qa  Alluvium (Holocene)", "#efe19a"),
    "Tmps": ("Tmps  Sentolo Fm. (Miocene–Pliocene)", "#c3e4ef"),
    "Tmph": ("Tmph  Halang Fm. (Miocene–Pliocene)", "#d3e8bb"),
    "Tmj": ("Tmj  Jonggrangan Fm. (Miocene)", "#7db7dc"),
    "Tomk": ("Tomk  Kebobutak Fm. (Oligocene–Miocene)", "#e6c29c"),
    "Teon": ("Teon  Nanggulan Fm. (Eocene–Oligocene)", "#b9a68b"),
    "da": ("da  Dacite (Miocene)", "#c58bd8"),
    "a1": ("a1  Andesite intrusions (Miocene)", "#d65d58"),
}


def load_geology():
    g = gpd.read_file(os.path.join(GEOLOGY_DIR, "Geologi.shp"), bbox=BBOX_LONLAT).to_crs(32749)
    f = gpd.read_file(os.path.join(GEOLOGY_DIR, "faultline.shp"), bbox=BBOX_LONLAT).to_crs(32749)
    for d in (g, f):
        for c in d.columns:
            if c != "geometry":
                d[c] = d[c].astype(object)
    assert set(g.DATA) == {"PUSLITBANG GEOLOGI"}
    return g, f


def geology_panel(ax, g, f, xy):
    W = box(*WINDOW)
    gw = g.clip(W)
    unknown = set(gw.SANDI) - set(UNITS)
    assert not unknown, unknown
    for code, (_, colour) in UNITS.items():
        part = gw[gw.SANDI == code]
        if len(part):
            part.plot(ax=ax, color=colour, edgecolor="white", linewidth=0.25, zorder=1)
    fw = f.clip(W)
    fw.plot(ax=ax, color=INK, linewidth=0.8, zorder=3)
    hull = np.vstack([convex_hull(xy), convex_hull(xy)[:1]])
    ax.plot(hull[:, 0], hull[:, 1], color=INK, lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c=INK, edgecolors="white", linewidths=0.4, zorder=5)
    gx, gy = GAJAH[0] * 1e3, GAJAH[1] * 1e3
    ax.scatter(gx, gy, marker="*", s=70, c=CAT[1], edgecolors="white", linewidths=0.6, zorder=6)
    ax.annotate("Gajah summit", (gx, gy), xytext=(5, 4), textcoords="offset points", fontsize=6.3, color=INK,
                va="bottom", zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_xlim(WINDOW[0], WINDOW[2])
    ax.set_ylim(WINDOW[1], WINDOW[3])
    ax.set_aspect("equal")
    ticks = np.arange(390000, 410001, 5000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t / 1e3:.0f}" for t in ticks])
    yt = np.arange(9130000, 9150001, 5000)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t / 1e3:.0f}" for t in yt])
    ax.set_xlabel("Easting (km, UTM zone 49S)")
    ax.set_ylabel("Northing (km)")
    ax.set_title("(a) Geology of the data window", loc="left", color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(True)
    return [code for code in UNITS if (gw.SANDI == code).any()]


def study_area_panel(fig, ax, cax):
    """Panel (b): as the original Figure 1 (make_new_figs.fig_study_area), adapted to the smaller panel."""
    x, y, z, _ = load_grid("../3_comparison_ggmplus/data/ggmplus_oasis.xls")
    st = pd.read_csv("../3_comparison_ggmplus/data/gravity_requested_area.csv")
    xy = st[["X", "Y"]].to_numpy()
    X, Y = x / 1e3, y / 1e3
    ls = LightSource(azdeg=315, altdeg=40)
    dx = np.median(np.diff(x, axis=1))
    hs = ls.hillshade(z, vert_exag=1.5, dx=dx, dy=dx)
    ax.pcolormesh(X, Y, np.where(z <= 0, np.nan, 1.0), cmap=ListedColormap(["#f3f2ec"]), shading="nearest", zorder=0)
    ax.pcolormesh(X, Y, np.ma.masked_where(z <= 0, hs), cmap="Greys_r", vmin=0.0, vmax=1.35, shading="nearest",
                  alpha=0.55, zorder=1)
    ax.contour(X, Y, z, levels=[250, 500, 750], colors=[MUTED], linewidths=0.3, zorder=2)
    hull = convex_hull(xy) / 1e3
    hull = np.vstack([hull, hull[:1]])
    ax.plot(hull[:, 0], hull[:, 1], color=INK, lw=0.7, ls=(0, (4, 2)), zorder=3)
    ax.add_patch(Rectangle((390, 9130), 20, 20, fill=False, ec=INK2, lw=0.8, zorder=3))
    ax.text(390.4, 9150.6, "A" if VS.VOLC else "(a)", fontsize=6.5, color=INK2, zorder=7)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ8)
    sc = ax.scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, c=st.CBA, cmap=cmap, vmin=100, vmax=150, s=13,
                    edgecolors=INK, linewidths=0.4, zorder=5)
    ax.scatter(*GAJAH, marker="*", s=60, c=CAT[1], edgecolors="white", linewidths=0.5, zorder=6)
    for q in MENOREH:
        ax.scatter(*q, marker="^", s=16, facecolors="white", edgecolors=INK, linewidths=0.6, zorder=6)
    ax.text(406.4, 9157.2, "Menoreh highlands\n(912–954 m)", ha="center", fontsize=6.0, color=INK, zorder=7)
    ax.text(381.0, 9128.3, "Indian Ocean", fontsize=6.3, color=MUTED, style="italic", zorder=7)
    ax.plot([413, 418], [9128.8, 9128.8], color=INK, lw=1.6, solid_capstyle="butt", zorder=7)
    ax.text(415.5, 9129.5, "5 km", ha="center", fontsize=6.3, color=INK, zorder=7)
    ax.annotate("", xy=(420.6, 9169.3), xytext=(420.6, 9166.3), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8))
    ax.text(420.6, 9165.2, "N", ha="center", va="top", fontsize=6.8, color=INK)
    ax.set_xlim(x.min() / 1e3, x.max() / 1e3)
    ax.set_ylim(y.min() / 1e3, y.max() / 1e3)
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("Easting (km, UTM zone 49S)")
    ax.set_title("(b) Study area and terrestrial CBA", loc="left", color=INK)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[100, 110, 120, 130, 140, 150])
    cb.set_label("Terrestrial complete Bouguer anomaly (mGal)", color=INK2, fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5)
    ins = ax.inset_axes([0.015, 0.74, 0.40, 0.245])
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world.plot(ax=ins, color="#e9e8e2", edgecolor=MUTED, linewidth=0.4)
    tr = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([x.min(), x.max(), x.max(), x.min(), x.min()], [y.min(), y.min(), y.max(), y.max(), y.min()])
    ins.fill(lon, lat, facecolor=CAT[1], edgecolor=CAT[1], lw=1.0, zorder=4)
    ins.text(108.0, -7.05, "Java", fontsize=5.8, color=INK2, ha="center", style="italic")
    ins.set_xlim(105.0, 115.0)
    ins.set_ylim(-9.0, -5.6)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_visible(True)
        sp.set_color(AXIS)
    ins.grid(False)
    return xy


def main():
    if not os.path.exists(os.path.join(GEOLOGY_DIR, "Geologi.shp")):
        raise SystemExit("Set GEOLOGY_DIR to the folder holding Geologi.shp and faultline.shp of the digital geological "
                         "map of Indonesia (Pusat Penelitian dan Pengembangan Geologi); the map is not redistributed.")
    g, f = load_geology()
    st = pd.read_csv("../3_comparison_ggmplus/data/gravity_requested_area.csv")
    xy = st[["X", "Y"]].to_numpy()
    fig = plt.figure(figsize=(6.68, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.30], width_ratios=[1.0, 1.0], hspace=0.30, wspace=0.18,
                          left=0.07, right=0.99, top=0.95, bottom=0.03)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    present = geology_panel(axa, g, f, xy)
    leg_ax = fig.add_subplot(gs[1, 0])
    leg_ax.axis("off")
    cb_area = fig.add_subplot(gs[1, 1])
    cb_area.axis("off")
    cax = cb_area.inset_axes([0.08, 0.80, 0.84, 0.10])
    study_area_panel(fig, axb, cax)
    handles = [Patch(facecolor=UNITS[c][1], edgecolor="none", label=UNITS[c][0]) for c in present]
    handles += [Line2D([], [], color=INK, lw=0.8, label="Fault (type not given in source)"),
                Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 2)), label="Convex hull of the stations"),
                Line2D([], [], marker="o", ls="none", ms=3, mfc=INK, mec="white", mew=0.4, label="Terrestrial station"),
                Line2D([], [], marker="*", ls="none", ms=7, mfc=CAT[1], mec="white", mew=0.5, label="Gajah summit")]
    leg_ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=5.6, handlelength=1.6, handleheight=0.9,
                  columnspacing=0.8, labelspacing=0.35, borderaxespad=0.0, bbox_to_anchor=(-0.08, 1.02))
    cb_area.text(0.5, 0.25, "Dashed line: convex hull of the 43 stations (345.6 km²); rectangle: data window "
                 + ("(A)." if VS.VOLC else "(a)."),
                 ha="center", va="center", fontsize=5.8, color=INK2, transform=cb_area.transAxes)
    save(fig, "fig01_study_area_geology")
    # summary of the units inside the station hull (for the text)
    H = MultiPoint([tuple(p) for p in xy]).convex_hull
    gh = g.clip(H)
    a = gh.assign(area=gh.area / 1e6).groupby("SANDI").area.sum()
    print("hull area %.1f km2" % (H.area / 1e6))
    print((a / a.sum() * 100).round(1).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
