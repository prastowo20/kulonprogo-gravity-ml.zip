"""Graphical abstract (1328 x 531 px aspect; written at 2656 x 1062 px).

All numbers are read from the result files of the analysis (no hand-typed values except labels).
Run from this folder after stages 2 and 3:  python make_graphical_abstract.py
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

matplotlib.use("Agg")
import logging  # noqa: E402
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import os  # noqa: E402
V1 = "../2_simulation_ml"
V2 = "../3_comparison_ggmplus/results"
FIG_OUT = os.environ.get("FIG_OUT", "../figures_manuscript")
os.makedirs(FIG_OUT, exist_ok=True)
OUT = os.path.join(FIG_OUT, "graphical_abstract")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2a9"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
GAJAH = (401.448, 9140.279)

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": "white",
})

# ------------------------------------------------------------------ data
z = np.load(f"{V1}/results/real_maps.npz")
cells, p = z["cells"], z["pA"]
st = pd.read_csv(f"{V1}/data/gravity_requested_area.csv")
peaks = pd.read_csv(f"{V1}/results/real_peaks.csv")
peaks = peaks[peaks.p >= 0.3].reset_index(drop=True)
gt = json.load(open(f"{V1}/results/global_test.json"))
ens = json.load(open(f"{V2}/ensemble_density.json"))
cmp = pd.read_csv(f"{V2}/model_comparison.csv").set_index("model")
leak = json.load(open(f"{V2}/leakage_summary.json"))
nv = pd.read_csv(f"{V2}/ggm_value.csv")
cv = pd.read_csv(f"{V2}/ggm_value_calibrated.csv")

gp = [v["global_p"] for v in ens.values() if isinstance(v, dict) and "global_p" in v]

fig = plt.figure(figsize=(13.28, 5.31), dpi=200)
fig.text(0.012, 0.955, "Eroded eruption centres from sparse gravity: simulation-trained ML, leakage-free tests and a "
         "calibrated check of GGMplus", fontsize=15.5, weight="bold", va="top")
fig.text(0.012, 0.885, "Kulon Progo dome, Java  ·  43 terrestrial stations  ·  synthetic scenes at the real station "
         "geometry  ·  nine learners  ·  scene-wise testing", fontsize=11.5, color=INK2, va="top")
COL = [0.012, 0.268, 0.537, 0.765]          # left edges of the four panel columns (figure fraction)
TY, CY = 0.805, 0.075                       # panel-title and caption baselines


def title(i, text):
    fig.text(COL[i], TY, text, fontsize=12.5, weight="bold", va="bottom")


def caption(i, text):
    fig.text(COL[i], CY, text, fontsize=10.5, color=INK2, va="top")


# ------------------------------------------------------------------ panel 1: map
ax = fig.add_axes([COL[0], 0.1, 0.215, 0.69])
xs, ys = np.unique(cells[:, 0]), np.unique(cells[:, 1])
Z = np.full((len(ys), len(xs)), np.nan)
Z[np.searchsorted(ys, cells[:, 1]), np.searchsorted(xs, cells[:, 0])] = p
lv = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
cmap = ListedColormap(SEQ)
ax.pcolormesh(xs / 1e3, ys / 1e3, Z, cmap=cmap, norm=BoundaryNorm(lv, cmap.N), shading="nearest")
ax.scatter(st.X / 1e3, st.Y / 1e3, s=9, c=INK, edgecolors="white", linewidths=0.4, zorder=4)
for k, r in peaks.iterrows():
    ax.scatter(r.x / 1e3, r.y / 1e3, marker="^", s=70, facecolors="white", edgecolors=INK, linewidths=1.2, zorder=5)
    ax.annotate(f"P{k + 1}", (r.x / 1e3, r.y / 1e3), xytext=(6, 2), textcoords="offset points", fontsize=11,
                weight="bold", zorder=6)
ax.scatter(*GAJAH, marker="*", s=150, c=ORANGE, edgecolors="white", linewidths=0.6, zorder=6)
ax.annotate("Gajah\nsummit", GAJAH, xytext=(7, -4), textcoords="offset points", fontsize=9.5, va="top", zorder=6)
ax.set_aspect("equal")
ax.set_xlim(390, 411)
ax.set_ylim(9129.5, 9151)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(False)
from matplotlib.cm import ScalarMappable  # noqa: E402
cax = fig.add_axes([COL[0] + 0.222, 0.33, 0.006, 0.3])
cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(lv, cmap.N), cmap=cmap), cax=cax, ticks=[0, 0.2, 0.4, 1.0])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8.5, length=2)
cb.ax.set_title("p", fontsize=9.5, color=INK2, pad=3)
title(0, "Candidate zones: not significant")
caption(0, f"max p = {p.max():.2f}; global test p = {min(gp):.2f}–{max(gp):.2f}\n"
        "no dense core beneath the Gajah summit")

# ------------------------------------------------------------------ panel 2: learners vs prior
ax = fig.add_axes([COL[1] + 0.093, 0.215, 0.158, 0.57])
order = ["HGB", "XGB", "RF", "ET", "MLP", "SVM", "LR", "DT", "GMM"]
lab = {"HGB": "Gradient boosting", "XGB": "XGBoost", "RF": "Random forest", "ET": "Extra trees", "MLP": "Neural net",
       "SVM": "SVM", "LR": "Logistic regr.", "DT": "Decision tree", "GMM": "Gaussian mixture"}
yy = np.arange(len(order))[::-1]
for y, m in zip(yy, order):
    ra, rb, rc = cmp.loc[m, "A_recall@1.0FP"], cmp.loc[m, "B_recall@1.0FP"], cmp.loc[m, "C_recall@1.0FP"]
    ax.plot([ra, rc], [y, y], color=GRID, lw=2, zorder=1)
    ax.scatter(rc, y, s=40, facecolors="white", edgecolors=GREEN, linewidths=1.1, zorder=3)
    ax.scatter(rb, y, s=40, facecolors="white", edgecolors=BLUE, linewidths=1.1, zorder=3)
    ax.scatter(ra, y, s=46, c=BLUE, zorder=4, edgecolors="white", linewidths=0.5)
ax.set_yticks(yy)
ax.set_yticklabels([lab[m] for m in order], fontsize=9.5)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlim(-0.02, 0.7)
ax.set_xlabel("Recall at ≤ 1 false peak per scene", fontsize=10)
ax.tick_params(axis="x", labelsize=9)
from matplotlib.lines import Line2D  # noqa: E402
hs = [Line2D([], [], ls="", marker="o", ms=6.5, mfc=BLUE, mec="white"),
      Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=BLUE),
      Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec=GREEN)]
ax.legend(hs, ["prior A", "B", "C"], loc="lower right", fontsize=9, frameon=True, facecolor="white",
          edgecolor="none", framealpha=1, handletextpad=0.1, borderaxespad=0.1, ncol=3, columnspacing=0.6)
ax.text(0.03, 0.5, "recall 0: maps without\ndistinct peaks", fontsize=8.5, color=MUTED, va="center")
title(1, "Best learners share one ceiling")
caption(1, "six of nine learners are equivalent;\nthe prior changes recall more")

# ------------------------------------------------------------------ panel 3: leakage
ax = fig.add_axes([COL[2] + 0.035, 0.215, 0.165, 0.57])
rnd, scn = leak["pooled"]["random cell split (leaky)"], leak["pooled"]["scene-wise split (honest)"]
vals = [(rnd["held_out_cells_AUC"], scn["held_out_cells_AUC"]), (rnd["new_maps_AUC"], scn["new_maps_AUC"])]
x = np.arange(2)
w = 0.36
b1 = ax.bar(x - w / 2, [v[0] for v in vals], w, color=ORANGE, label="random cell split")
b2 = ax.bar(x + w / 2, [v[1] for v in vals], w, color=BLUE, label="scene-wise split")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height():.2f}", ha="center",
                fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(["held-out cells", "new maps"], fontsize=10)
ax.set_ylim(0.5, 1.08)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("AUC", fontsize=10)
ax.tick_params(axis="y", labelsize=9)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=9, frameon=False, handlelength=1, borderaxespad=0.0)
title(2, "Random splits inflate skill")
caption(2, "only unseen scenes\nmeasure transfer")

# ------------------------------------------------------------------ panel 4: GGMplus value
ax = fig.add_axes([COL[3], 0.215, 0.215, 0.57])
base = nv[nv.model == "PSG only"].iloc[0]["recall@1.0FP"]
naive = nv[nv.ggm_err_mGal == 0.0]
cal = cv[cv.variant == "calibrated_best_fit_short"]
rows = [("Terrestrial only", base, GREY, 1.0),
        ("Terrestrial + GGMplus, naive emulation", naive[naive.model == "PSG + GGM"].iloc[0]["recall@1.0FP"], BLUE, 0.5),
        ("GGMplus only, naive emulation", naive[naive.model != "PSG + GGM"].iloc[0]["recall@1.0FP"], ORANGE, 0.5),
        ("Terrestrial + GGMplus, calibrated", cal[cal.model == "PSG + GGM"].iloc[0]["recall@1.0FP"], BLUE, 1.0),
        ("GGMplus only, calibrated", cal[cal.model != "PSG + GGM"].iloc[0]["recall@1.0FP"], ORANGE, 1.0)]
yy = np.arange(len(rows))[::-1] * 1.0
for y, (nm, v, c, al) in zip(yy, rows):
    ax.barh(y - 0.12, v, 0.42, color=c, alpha=al)
    ax.text(v + 0.012, y - 0.12, f"{v:.2f}", va="center", fontsize=10, zorder=5,
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.text(0.003, y + 0.2, nm, va="bottom", fontsize=9.5, color=INK2)
ax.axvline(base, color=INK, lw=1, zorder=0)
ax.set_yticks([])
ax.set_ylim(-0.5, len(rows) - 0.35)
ax.set_xlim(0, 0.68)
ax.set_xlabel("Recall at ≤ 1 false peak per scene", fontsize=10)
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="y", visible=False)
ax.spines["left"].set_visible(False)
title(3, "GGMplus adds no skill")
caption(3, "a naive emulation leaks\ntarget-scale signal")

fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
from PIL import Image  # noqa: E402
print("written", OUT + ".png", Image.open(OUT + ".png").size)
