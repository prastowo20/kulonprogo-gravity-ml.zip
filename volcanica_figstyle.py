"""Volcanica house style for figure panels (Word template, October 2026).

The Volcanica template asks that separate panels be labelled with an upper-case letter in a 12-point sans-serif
font, and that captions refer to them as [A], [B], ... This module is switched on with the environment variable
PANEL_STYLE=volcanica. The figure scripts then
- call relabel(fig) just before saving: a leading panel token in an axes title, "(a) " or "a  ", is removed and the
  upper-case letter is drawn in bold 12 pt at the start of the title, which moves right to make room; the letter
  size is corrected for the later reduction of figures wider than 175 mm (85 mm for one-column figures), so that
  it is 12 pt in print;
- write to FIG_OUT_VOLCANICA (default $FIG_OUT/volcanica, i.e. figures_manuscript/volcanica) instead of the usual
  folder.
tok(i) returns the token for panel i (0 -> "a  ") in Volcanica mode and "" otherwise, for figures whose panels had
no letters before; si(s) converts the density units g/cm³ of figure labels to kg m⁻³ in Volcanica mode.
"""
from __future__ import annotations

import os
import re

VOLC = os.environ.get("PANEL_STYLE", "") == "volcanica"
OUT = os.environ.get("FIG_OUT_VOLCANICA", os.path.join(os.environ.get("FIG_OUT", "../figures_manuscript"), "volcanica"))
LETTER_PT = 12.0
ONE_COL_MM, TWO_COL_MM = 85.0, 175.0
TOKEN = re.compile(r"^(?:\(([a-z])\) +|([a-z]) {2,})")


def tok(i):
    return f"{'abcdefghijklmnopqrstuvwxyz'[i]}  " if VOLC else ""


SI = {"2.84 → 2.67 g/cm³": "2840 → 2670 kg m⁻³", "Δρ = +0.3 g/cm³": "Δρ = +300 kg m⁻³"}


def si(s):
    """Density units in SI (kg m⁻³), as in the Volcanica text, in Volcanica mode."""
    if VOLC:
        for k, v in SI.items():
            s = s.replace(k, v)
    return s


def out_dir(default):
    if VOLC:
        os.makedirs(OUT, exist_ok=True)
        return OUT
    return default


def _print_scale(fig, renderer):
    """Factor by which the exported figure is reduced (export_figures_volcanica.py: widths > 90 mm go to 175 mm,
    narrower ones to 85 mm; figures are never enlarged)."""
    w_mm = fig.get_tightbbox(renderer).width * 25.4
    target = ONE_COL_MM if w_mm <= 90.0 else TWO_COL_MM
    return min(1.0, target / w_mm)


def relabel(fig, gap_pt=3.0, weight="bold"):
    """Replace leading panel tokens in axes titles by 12-pt upper-case letters (Volcanica style)."""
    fig.canvas.draw()
    found = []
    for ax in fig.axes:
        for t in (ax._left_title, ax.title, ax._right_title):
            s = t.get_text()
            m = TOKEN.match(s)
            if m:
                found.append((ax, t, (m.group(1) or m.group(2)).upper()))
                t.set_text(s[m.end():])
    if not found:
        return 0
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    size = LETTER_PT / _print_scale(fig, renderer)
    if fig.get_layout_engine() is not None:          # freeze the layout so that positions stay where they were drawn
        fig.set_layout_engine("none")
    family = fig.axes[0].title.get_fontfamily()
    for ax, t, letter in found:
        ab = ax.get_window_extent(renderer)
        tb = t.get_window_extent(renderer) if t.get_text() else None
        top = tb.y1 if tb is not None else ab.y1 + 4.0 * fig.dpi / 72.0
        lab = fig.text(ab.x0 / fig.bbox.width, top / fig.bbox.height, letter, ha="left", va="top", fontsize=size,
                       fontweight=weight, family=family, color=t.get_color())
        if t is ax._left_title and t.get_text():     # move the left title to the right of the letter
            lw = lab.get_window_extent(renderer).width + gap_pt * fig.dpi / 72.0
            x, y = t.get_position()
            t.set_position((x + lw / ab.width, y))
    return len(found)
