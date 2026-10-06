"""Figure 10 (revision, 30 Sep 2026): real probability maps paired with their null benchmark.

Row 1: probability maps of the real terrestrial data (H1) under priors A (reference ensemble), B and C.
Row 2: null benchmark for each prior - distribution of the map maximum in 1000 target-free scenes
       (3_comparison_ggmplus/results/null_benchmark.npz, regenerated with the seeds of global_test.py and checked
       against the stored summaries), with the real map maximum and the global p-value.
Row 3: local uncertainty layers of the reference map (as in the original Figure 10).
Option --volcanica: smaller canvas so that the figure fits the Volcanica limit (175 x 200 mm);
       written with the suffix _volcanica. The default size is used by the Indonesian manuscript.
       Panel (a) of the Volcanica version also outlines the mapped andesite intrusions and dacite inside the
       station hull (revision of 5 Oct 2026; geological map read from GEOLOGY_DIR, not redistributed).
Output: $FIG_OUT/fig10_real_map_null_uncertainty.{png,pdf} (default ../figures_manuscript)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plots_ml_ms as P
import volcanica_figstyle as VS  # path added by plots_ml_ms (PANEL_STYLE=volcanica: upper-case panel letters)
from features import FeatureExtractor
from geometry import convex_hull, load_psg
from priors_cfg import REFERENCE, SCENARIOS

NULL = os.environ.get("NULL_BENCHMARK", "../3_comparison_ggmplus/results/null_benchmark")
OUT = os.path.join(os.environ.get("FIG_OUT", "../figures_manuscript"), "fig10_real_map_null_uncertainty")


def intrusion_outlines(ax, xy):
    """Outlines of the andesite intrusions (a1) and dacite (da) of the digital geological map of Indonesia
    (Pusat Penelitian dan Pengembangan Geologi), drawn only inside the station hull where the map is defined;
    data not redistributed, folder given by GEOLOGY_DIR."""
    import os

    import geopandas as gpd
    from shapely.geometry import MultiPoint

    d = os.environ.get("GEOLOGY_DIR", "")
    if not os.path.exists(os.path.join(d, "Geologi.shp")):
        print("GEOLOGY_DIR not set or Geologi.shp not found: intrusion outlines in panel (a) skipped")
        return
    g = gpd.read_file(os.path.join(d, "Geologi.shp"), bbox=(109.94, -7.93, 110.24, -7.63)).to_crs(32749)
    g["SANDI"] = g["SANDI"].astype(object)
    hull = MultiPoint([tuple(q) for q in xy]).convex_hull
    for geom in g[g.SANDI.isin(["a1", "da"])].dissolve(by="SANDI").geometry:
        lines = geom.boundary.intersection(hull)
        for line in getattr(lines, "geoms", [lines]):
            if line.geom_type != "LineString" or line.is_empty:
                continue
            x, y = line.xy
            ax.plot(np.asarray(x) / 1e3, np.asarray(y) / 1e3, color=P.INK, lw=0.55, alpha=0.85, zorder=3)


def main(volcanica=False):
    xy, _ = load_psg("data/gravity_requested_area.csv")
    fe = FeatureExtractor(xy, convex_hull(xy))
    z = np.load("results/real_maps.npz")
    pk = pd.read_csv("results/real_peaks.csv")
    pk = pk[pk.p >= 0.3]
    nz = np.load(NULL + ".npz")
    nj = json.load(open(NULL + ".json"))
    assert all(all(v.values()) if isinstance(v, dict) else v for v in nj["reproduction_checks"].values())

    out = OUT + ("_volcanica" if volcanica else "")
    fig = plt.figure(figsize=((6.45, 7.6) if VS.VOLC else (6.45, 7.75)) if volcanica else (7.2, 8.9), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.52, 1.0])
    top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    mid = [fig.add_subplot(gs[1, j]) for j in range(3)]
    bot = [fig.add_subplot(gs[2, j]) for j in range(3)]

    lv_p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    letters = "abc"
    for ax, name, L in zip(top, SCENARIOS, letters):
        v = z["pA"] if name == REFERENCE else z[f"p_{name}"]
        s = nj["null"][f"{name} | H1"]
        rel = "<" if s["real_max_p"] < s["null_p95"] else "≥"
        sub = f"max {s['real_max_p']:.2f} {rel} null P95 {s['null_p95']:.2f}; global p {s['global_p']:.3f}"
        if VS.VOLC:   # Volcanica panel letters take room at the start of the title: statistics on two short lines
            sub = f"max {s['real_max_p']:.2f} {rel} null P95 {s['null_p95']:.2f}\nglobal p {s['global_p']:.3f}"
        cf = P.map_panel(ax, fe, v, lv_p, f"({L}) {'Prior ' if VS.VOLC else ''}{P.SCEN_LABEL[name]}\n{sub}", xy=xy,
                         peaks=pk if name == REFERENCE else None, label_gajah=name == REFERENCE and not volcanica)
        if volcanica and name == REFERENCE:   # mapped intrusions for comparison with P1 and P2 (Figure 1a)
            intrusion_outlines(ax, xy)
        if volcanica and name == REFERENCE:   # label on a light box so that no station dot overprints it
            ax.annotate("Gajah summit", P.GAJAH, xytext=(4, 4), textcoords="offset points", fontsize=6.3,
                        color=P.INK, va="bottom", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    P.row_colorbar(fig, cf, top, "Probability of a complex-scale centre zone")

    bins = np.arange(0.0, 1.0001, 0.025)
    short = {"A_clutter": "A", "B_mixed": "B", "C_body": "C"}
    for k, (ax, name, L) in enumerate(zip(mid, SCENARIOS, "def")):
        mx0 = nz[f"null_max_{name}"]
        s = nj["null"][f"{name} | H1"]
        lab = (("Null scenes", "Null P95", "Real max") if volcanica
               else ("Target-free scenes", "Null 95th pct", "Real map max"))
        cnt, _, _ = ax.hist(mx0, bins=bins, color=P.SEQ[1], edgecolor="white", linewidth=0.4, label=lab[0])
        ax.axvline(s["null_p95"], color=P.INK2, lw=0.9, ls=(0, (3, 2)), label=lab[1])
        ax.axvline(s["real_max_p"], color=P.CAT[1], lw=1.6, label=lab[2])
        if volcanica:      # shorter lines so that the box clears the vertical lines of the narrower panels
            stats = (f"real max {s['real_max_p']:.2f}\nnull median {s['null_median']:.2f}\n"
                     f"null P95 {s['null_p95']:.2f}\nglobal p {s['global_p']:.3f}")
        else:
            stats = (f"real max = {s['real_max_p']:.2f}\nnull median = {s['null_median']:.2f}\n"
                     f"null 95th pct = {s['null_p95']:.2f}\nglobal p = {s['global_p']:.3f}")
        ax.text(0.03, 0.96, stats,
                transform=ax.transAxes, ha="left", va="top", fontsize=5.9 if volcanica else 6.4, color=P.INK,
                linespacing=1.3, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, cnt.max() * 1.75)
        ax.set_title(f"({L}) Null benchmark, prior {short[name]}", loc="left", color=P.INK)
        ax.set_xlabel("Maximum probability of the map")
        if k == 0:
            ax.set_ylabel("Number of scenes")
            ax.legend(loc="upper right", bbox_to_anchor=(1.03, 1.0) if volcanica else (1.06, 1.0),
                      fontsize=5.6 if volcanica else 6.0, handlelength=1.2 if volcanica else 1.4, borderaxespad=0.2)
        ax.grid(axis="x", visible=False)

    lv_u = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]
    P.map_panel(bot[0], fe, z["prior_range"], lv_u, "(g) Prior spread (max − min, A–C)")
    P.map_panel(bot[1], fe, z["noise_sd"], lv_u, "(h) Data-noise SD (σ = 2 mGal)", xy=xy)
    cf = P.map_panel(bot[2], fe, z["jk_se"], lv_u, "(i) Station-jackknife SE", xy=xy)
    P.row_colorbar(fig, cf, bot, "Uncertainty (probability units)")
    for ax in bot:
        ax.set_xlabel("Easting (km)")
    for ax in (top[0], bot[0]):
        ax.set_ylabel("Northing (km)")
    if VS.VOLC:   # Volcanica template: upper-case 12 pt panel letters, written to the Volcanica figure folder
        VS.relabel(fig)
        out = os.path.join(VS.out_dir(os.path.dirname(OUT)), os.path.basename(out))
    fig.savefig(out + ".png", bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--volcanica", action="store_true", help="175 x 200 mm version for Volcanica")
    main(ap.parse_args().volcanica)
