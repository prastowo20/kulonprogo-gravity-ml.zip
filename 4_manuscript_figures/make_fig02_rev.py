"""Figure 2 (workflow), revision of 30 Sep 2026.

Same three strands as the original Figure 2, but the detectability limit is now the first step of
strand 2, validation of the simulations against the real data is drawn as a gate before training,
and every real probability map is paired with its null benchmark. Core steps are outlined in blue.
Option --volcanica: SI density units and "detection limit" wording, written with the suffix _volcanica
(the default version is used by the Indonesian manuscript).
Output: $FIG_OUT/fig02_workflow_rev.{png,pdf} (default ../figures_manuscript)
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_new_figs import AXIS, BAND, CAT, INK, INK2, arrow, save

CORE_FC = "#e6f0fc"


def box(ax, x, y, w, h, text, tag, core=False, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                fc=CORE_FC if core else BAND, ec=CAT[0] if core else AXIS, lw=1.3 if core else 0.6))
    ax.text(x + w / 2, y + h / 2 - 0.016, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.22)
    ax.text(x + 0.008, y + h - 0.01, tag, ha="left", va="top", fontsize=6.3, color=CAT[0] if core else INK2,
            fontweight="bold")



def _tr_lines(text, tr):
    """Apply the Volcanica wording to a multi-line box text, first as a whole, then line by line."""
    if tr(text) != text:
        return tr(text)
    out = text
    for k in VOLCANICA:
        if "\n" not in k and k in out:
            out = out.replace(k, tr(k))
    return out


VOLCANICA = {  # wording and SI units of the Volcanica manuscript
    "2  Detectability and detection (RQ2)": "2  Detection limit and detection (RQ2)",
    "Detectability limit (first step)": "Detection limit (first step)",
    "(2.67 g/cm³, full tile)": "(2670 kg m⁻³, full tile)",
    "H2: 2.84 → 2.67 g/cm³": "H2: 2840 → 2670 kg m⁻³",
    "G = a · LP_λ(true field) + errors": r"G = a · LP$_\lambda$(true field) + errors",
    "Validation against the real data\n(gate before training): 10 statistics,\npriors A, B, C, 5–95% band":
        "Validation against the real data\n(gate for the reference prior):\n10 statistics, priors A–C, 5–95% band",
    "Blue outline: core steps for an honest detectability statement – detectability limit (2a), ":
        "Blue outline: core steps for an honest detection statement – detection limit (2a), ",
}


def fig_workflow_rev(volcanica=False):
    tr = (lambda t: VOLCANICA.get(t, t)) if volcanica else (lambda t: t)
    fig, ax = plt.subplots(figsize=(7.2, 4.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    W, H = 0.31, 0.15
    X = [0.01, 0.345, 0.68]
    heads = ["1  Data and GGMplus reliability (RQ1)", "2  Detectability and detection (RQ2)",
             "3  GGMplus value and inference (RQ2, RQ3)"]
    rows = [0.795, 0.615, 0.435, 0.255, 0.075]
    cols = [[("Terrestrial CBA\n43 stations, median spacing 2.4 km", False),
             ("GGMplus grid (220 m)\ngravity and surface height", False),
             ("Re-reduction of GGMplus\nfree-air anomaly − prism topographic\neffect (2.67 g/cm³, full tile)", False),
             ("Band-wise comparison\nall λ; λ > 10 km; λ > 18.3 km;\n2nd-order trend and residual", False),
             ("Reduction-density hypotheses\nH1: CBA as supplied\nH2: 2.84 → 2.67 g/cm³", False)],
            [("Detectability limit (first step)\nmatched filter D1, map reading D2;\nR80 vs station spacing and noise", True),
             ("Scene generator at the real geometry\ntargets R = 1.5–4 km (from 2a),\ndistractors, regional field, noise", False),
             ("Validation against the real data\n(gate before training): 10 statistics,\npriors A, B, C, 5–95% band", True),
             ("Scene-wise holdout, no shared scenes\n4000 train / 800 validation / 1000 test;\n26 features; nine learners as control", True),
             ("Object-level metrics (FROC), cell\nmetrics, paired scene bootstrap,\nleakage experiment", False)],
            [("GGMplus emulation\nG = a · LP_λ(true field) + errors\n(144 variants)", False),
             ("Misfit check against 1d\n11 statistics of the real\nterrestrial–GGMplus comparison", False),
             ("Value of GGMplus features\npaired differences and\n'GGMplus only' leakage diagnostic", False),
             ("Real data: each probability map\npaired with its null benchmark\n(1000 target-free scenes, global p)", True),
             ("Local uncertainty: ensemble, noise,\nstation jackknife, prior, density (1e);\ncovariate-shift check", False)]]
    for j, (x0, head, items) in enumerate(zip(X, heads, cols)):
        ax.text(x0, 0.985, tr(head), fontsize=7.2, fontweight="bold", color=INK, va="top")
        for i, (yy, (t, core)) in enumerate(zip(rows, items)):
            t = _tr_lines(t, tr) if volcanica else t
            box(ax, x0, yy, W, H, t, f"{j + 1}{'abcde'[i]}", core=core)
            if i:
                arrow(ax, (x0 + W / 2, rows[i - 1]), (x0 + W / 2, yy + H))
    arrow(ax, (X[0] + W, rows[0] + H / 2), (X[1], rows[0] + H / 2))      # station geometry -> detectability
    arrow(ax, (X[1] + W, rows[1] + H / 2), (X[2], rows[0] + H / 2))      # scenes -> GGMplus emulation
    arrow(ax, (X[1] + W, rows[3] + H / 2), (X[2], rows[3] + H / 2))      # trained models -> real data
    ax.text(0.01, 0.022, tr("Blue outline: core steps for an honest detectability statement – detectability limit (2a), ")
            + "validation gate (2c), leakage-free holdout (2d) and null benchmark (3d).",
            fontsize=6.4, color=INK2, va="bottom")
    save(fig, "fig02_workflow_rev" + ("_volcanica" if volcanica else ""))


if __name__ == "__main__":
    fig_workflow_rev(volcanica="--volcanica" in sys.argv[1:])
    print("done")
