#!/usr/bin/env python3
"""Draw where Kefir 1's parameters are: all parameters against the ones active per token.

Counts come from the model definition (see docs/architecture.md, "Where the parameters are"); writes
results/figures/architecture_params.{svg,png} and architecture_params-dark.png.

    python results/make_param_breakdown.py
"""
from __future__ import annotations

import matplotlib

import make_figures
from make_figures import save

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPERT = 3 * 256 * 352  # one SwiGLU expert
PARTS = [  # (name, short name, all parameters, active per token, colour, text colour)
    ("Routed experts", "Experts", 64 * EXPERT * 8, 2 * EXPERT * 8, "#009e73", "#ffffff"),
    ("GDN-2 mixers", "GDN-2", 2_774_424, 2_774_424, "#0072b2", "#ffffff"),
    ("Attention mixers", "Attn", 655_616, 655_616, "#d55e00", "#ffffff"),
    ("Tied embedding", "Embedding", 2_097_152, 2_097_152, "#c8d2da", "#182b3a"),
    ("Routers and norms", "", 131_072 + 4_352, 131_072 + 4_352, "#52616e", "#ffffff"),
]
TOTAL = sum(p[2] for p in PARTS)
ACTIVE = sum(p[3] for p in PARTS)
assert (TOTAL, ACTIVE) == (144_074_648, 9_987_992)


def draw() -> None:
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11})
    fig, ax = plt.subplots(figsize=(6.6, 2.3))
    fig.subplots_adjust(left=.02, right=.98, top=.98, bottom=.02)
    rows = [("All parameters", TOTAL, 2), ("Active per token", ACTIVE, 3)]
    for y, (label, total, idx) in enumerate(rows):
        left = 0.0
        for part in PARTS:
            name, short, colour, tcolour = part[0], part[1], part[4], part[5]
            value = part[idx]
            share = value / total
            ax.barh([y], [share], left=left, height=.5, color=colour, linewidth=0)
            if share > .09:
                text = f"{name if share > .5 else short}  {value / 1e6:.1f}M"
                ax.text(left + share / 2, y, text, ha="center", va="center", color=tcolour, fontsize=9.5)
            elif share > .05:
                ax.text(left + share / 2, y, f"{short}\n{value / 1e6:.1f}M", ha="center", va="center",
                        color=tcolour, fontsize=8.5, linespacing=1.0)
            left += share
        ax.text(0, y - .33, f"{label}  {total / 1e6:.1f}M", ha="left", va="bottom", fontsize=11,
                color=make_figures.INK, weight="bold")
    other = TOTAL - PARTS[0][2]
    ax.text(1, -.33, f"everything else {other / 1e6:.1f}M", ha="right", va="bottom", fontsize=9.5,
            color=make_figures.MUTED)
    ax.text(1, 1 - .33, "bars show the share of each total", ha="right", va="bottom", fontsize=9.5,
            color=make_figures.MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.35, -.75)
    ax.axis("off")
    save(fig, "architecture_params", release=True)


if __name__ == "__main__":
    draw()
