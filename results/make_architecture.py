#!/usr/bin/env python3
"""Draw the Kefir 1 architecture diagram from results/data/model.json.

Writes results/figures/architecture.svg (text as paths) and architecture.png (300 dpi).
Deterministic for a fixed matplotlib version.

    python results/make_architecture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

from make_figures import save

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Okabe-Ito colour-blind-safe palette & typography matching results/make_figures.py
INK = "#182b3a"
MUTED = "#52616e"
BORDER = "#a9b5bd"
FRAME_EDGE = "#c3cdd4"
SUBTLE_BG = "#f7f9fb"
INACTIVE_FILL = "#edf2f6"
INACTIVE_EDGE = "#c8d2da"

BLUE = "#0072B2"       # GDN-2
BLUE_FILL = "#eaf3fa"
AMBER = "#D55E00"      # Causal attention (Okabe-Ito vermilion)
AMBER_FILL = "#fdf0e8"
GREEN = "#009E73"      # Active MoE experts & routing
GREEN_FILL = "#e6f6f1"

plt.rcParams.update({
    "svg.hashsalt": "kefir-results",
    "svg.fonttype": "path",
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "text.color": INK,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fc: str = "white",
    ec: str = BORDER,
    lw: float = 0.85,
    r: float = 0.10,
    ls: str = "-",
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc,
        ec=ec,
        lw=lw,
        ls=ls,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = MUTED,
    lw: float = 1.0,
    head_len: float = 4.0,
    head_wid: float = 2.5,
    rad: float = 0.0,
    casing_color: str | None = None,
    zorder: int = 3,
) -> None:
    p = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=f"-|>,head_length={head_len},head_width={head_wid}",
        mutation_scale=1.0,
        color=color,
        lw=lw,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    if casing_color is not None:
        p.set_path_effects([pe.Stroke(linewidth=lw + 2.2, foreground=casing_color), pe.Normal()])
    ax.add_patch(p)


def plus_node(ax, cx: float, cy: float, radius: float = 0.165, zorder: int = 4) -> None:
    ax.add_patch(Circle((cx, cy), radius, fc="white", ec=MUTED, lw=0.95, zorder=zorder))
    ax.text(cx, cy - 0.008, "+", ha="center", va="center", fontsize=11, weight="bold", color=INK, zorder=zorder + 1)


def draw_architecture() -> None:
    m = json.loads((DATA / "model.json").read_text())

    n_layers = int(m["n_layers"])
    pattern = list(m["layer_pattern"])
    layers = (pattern * (n_layers // len(pattern)))[:n_layers]
    d_model = int(m["d_model"])
    n_heads = int(m["attention_heads"])
    head_dim = int(m["head_dim"])
    chunk_size = int(m["gdn_chunk_size"])
    num_experts = int(m["num_experts"])
    top_k = int(m["top_k"])
    expert_d_ff = int(m["expert_d_ff"])
    vocab_sem = int(m["vocab_semantic"])
    vocab_pad = int(m["vocab_padded"])
    ctx_len = int(m["context_length"])
    p_total = int(m["parameters_total"])
    p_active = int(m["parameters_active_per_token"])

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.03)
    ax.set(xlim=(0, 14.0), ylim=(0, 8.55))
    ax.set_aspect("equal")
    ax.axis("off")

    # Header matching results/make_figures.py
    fig.text(0.045, 0.955, "Kefir 1 architecture", fontsize=15, weight="bold", va="top", color=INK)
    fig.text(
        0.045,
        0.898,
        f"{p_total:,} parameters · {p_active:,} active per token · {ctx_len:,}-token context · width {d_model}",
        fontsize=10,
        color=MUTED,
        va="top",
    )

    # =========================================================================
    # LEFT PANEL: Model stack (bottom-to-top)
    # =========================================================================
    lx, lw_box = 1.45, 3.25
    lcx = lx + lw_box / 2

    ax.text(lx, 8.15, "Model", fontsize=11, weight="bold", color=INK, va="center")

    # Bottom input label
    ax.text(lcx, 0.28, f"Token IDs (≤ {ctx_len:,})", ha="center", va="center", fontsize=8.8, color=MUTED)
    arrow(ax, lcx, 0.45, lcx, 0.86)

    # Token embedding
    emb_y, emb_h = 0.88, 0.68
    round_box(ax, lx, emb_y, lw_box, emb_h, fc=SUBTLE_BG, ec=BORDER, lw=0.9, r=0.10)
    ax.text(lcx, emb_y + emb_h * 0.63, "Token embedding", ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(
        lcx,
        emb_y + emb_h * 0.27,
        f"{vocab_pad:,} rows × {d_model}",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )
    arrow(ax, lcx, emb_y + emb_h, lcx, 1.96)

    # 8 Decoder layers in (3 × GDN-2 + 1 × Attention) × 2
    blk_h = 0.35
    gap_in = 0.06
    gap_cycle = 0.24
    y_cur = 1.98
    layer_boxes: list[tuple[float, float, str]] = []
    for idx, kind in enumerate(layers):
        is_mha = kind == "mha"
        fc = AMBER_FILL if is_mha else BLUE_FILL
        ec = AMBER if is_mha else BLUE
        label = "Causal attention" if is_mha else "GDN-2"
        round_box(ax, lx, y_cur, lw_box, blk_h, fc=fc, ec=ec, lw=0.95, r=0.07)
        ax.text(lx + 0.16, y_cur + blk_h / 2, f"{idx + 1}", ha="left", va="center", fontsize=7.9, color=MUTED)
        ax.text(lcx + 0.06, y_cur + blk_h / 2, label, ha="center", va="center", fontsize=9.0, weight="bold", color=ec)
        layer_boxes.append((y_cur, y_cur + blk_h, kind))
        if idx == 3:
            arrow(ax, lcx, y_cur + blk_h, lcx, y_cur + blk_h + gap_cycle - 0.02, lw=0.9, head_len=3.4, head_wid=2.1)
            y_cur += blk_h + gap_cycle
        else:
            y_cur += blk_h + gap_in

    # Cycle brackets on the left of layers 1-4 and 5-8
    for start_i, end_i in [(0, 3), (4, 7)]:
        by0 = layer_boxes[start_i][0] + 0.03
        by1 = layer_boxes[end_i][1] - 0.03
        bx = lx - 0.14
        ax.plot([bx + 0.07, bx, bx, bx + 0.07], [by0, by0, by1, by1], color=MUTED, lw=0.8)
        ax.text(bx - 0.07, (by0 + by1) / 2, "3:1", ha="right", va="center", fontsize=8.0, color=MUTED)

    # Final RMSNorm
    stack_top = layer_boxes[-1][1]
    fn_y, fn_h = stack_top + 0.34, 0.44
    arrow(ax, lcx, stack_top, lcx, fn_y - 0.02)
    round_box(ax, lx + 0.25, fn_y, lw_box - 0.50, fn_h, fc=SUBTLE_BG, ec=BORDER, lw=0.9, r=0.10)
    ax.text(lcx, fn_y + fn_h / 2, "Final RMSNorm", ha="center", va="center", fontsize=9.0, weight="bold")

    # Tied output head
    head_y, head_h = fn_y + fn_h + 0.32, 0.68
    arrow(ax, lcx, fn_y + fn_h, lcx, head_y - 0.02)
    round_box(ax, lx, head_y, lw_box, head_h, fc=SUBTLE_BG, ec=BORDER, lw=0.9, r=0.10)
    ax.text(lcx, head_y + head_h * 0.63, "Tied output head", ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(
        lcx,
        head_y + head_h * 0.27,
        f"{vocab_sem:,} semantic IDs",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )
    arrow(ax, lcx, head_y + head_h, lcx, head_y + head_h + 0.36)
    ax.text(lcx, head_y + head_h + 0.52, "Next-token logits", ha="center", va="center", fontsize=8.8, color=MUTED)

    # Weight-tying dashed connector on far left
    tx = 0.52
    emb_mid = emb_y + emb_h / 2
    head_mid = head_y + head_h / 2
    ax.plot([lx, tx, tx, lx], [emb_mid, emb_mid, head_mid, head_mid], color=MUTED, ls="--", lw=0.85, zorder=1)
    ax.text(
        tx,
        (emb_mid + head_mid) / 2,
        "weight-tied",
        ha="center",
        va="center",
        fontsize=7.9,
        color=MUTED,
        rotation=90,
        bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none"),
        zorder=2,
    )

    # =========================================================================
    # ZOOM CONNECTOR: Layer 5 -> Expanded Decoder Block
    # =========================================================================
    rx, ry, rw, rh = 5.65, 0.44, 7.90, 7.16
    zy0, zy1, _ = layer_boxes[4]  # Layer 5 (representative GDN-2 layer)
    ax.plot([lx + lw_box, rx], [zy1, ry + rh - 0.16], color=BORDER, ls=":", lw=1.0, zorder=1)
    ax.plot([lx + lw_box, rx], [zy0, ry + 0.16], color=BORDER, ls=":", lw=1.0, zorder=1)

    # =========================================================================
    # RIGHT PANEL: Expanded Decoder Layer (bottom-to-top)
    # =========================================================================
    ax.text(rx, 8.15, "Decoder layer", fontsize=11, weight="bold", color=INK, va="center")
    ax.text(
        rx + rw,
        8.15,
        f"× {n_layers}  ·  width {d_model}",
        fontsize=9.0,
        color=MUTED,
        ha="right",
        va="center",
    )

    round_box(ax, rx, ry, rw, rh, fc=SUBTLE_BG, ec=FRAME_EDGE, lw=0.95, r=0.16, zorder=1)

    # Main data trunk x-center and right-side residual bypass x
    tcx = 9.20
    res_x = 12.90

    # Entry arrow into decoder layer
    arrow(ax, tcx, 0.14, tcx, 0.86)

    # ---- Stage 1: Norm -> Token Mixer -> Add ----
    split1_y = 0.65
    add1_y = 2.74
    ax.plot([tcx, res_x, res_x], [split1_y, split1_y, add1_y], color=MUTED, lw=0.95, zorder=2)
    arrow(ax, res_x, add1_y, tcx + 0.185, add1_y, lw=0.95)
    ax.scatter([tcx], [split1_y], s=14, color=MUTED, zorder=4)

    # RMSNorm 1
    n1_y, n1_h, n1_w = 0.88, 0.40, 2.75
    round_box(ax, tcx - n1_w / 2, n1_y, n1_w, n1_h, fc="white", ec=BORDER, lw=0.9, r=0.10)
    ax.text(tcx, n1_y + n1_h / 2, "RMSNorm", ha="center", va="center", fontsize=9.1, weight="bold")
    arrow(ax, tcx, n1_y + n1_h, tcx, 1.48)

    # Token Mixer container with GDN-2 vs Causal Attention split
    tm_y, tm_h, tm_w = 1.50, 0.94, 5.30
    round_box(ax, tcx - tm_w / 2, tm_y, tm_w, tm_h, fc="white", ec=BORDER, lw=0.9, r=0.11)
    ax.text(
        tcx,
        tm_y + tm_h - 0.17,
        f"Token mixer  ·  {n_heads} heads × {head_dim}",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )

    pill_h = 0.43
    pill_y = tm_y + 0.13
    # GDN-2 pill (left)
    round_box(ax, tcx - tm_w / 2 + 0.18, pill_y, 2.30, pill_h, fc=BLUE_FILL, ec=BLUE, lw=0.95, r=0.08)
    ax.text(
        tcx - tm_w / 2 + 0.18 + 1.15,
        pill_y + pill_h / 2,
        f"GDN-2 · chunk {chunk_size}",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        color=BLUE,
    )
    ax.text(tcx, pill_y + pill_h / 2, "or", ha="center", va="center", fontsize=8.2, color=MUTED)
    # Causal attention pill (right)
    round_box(ax, tcx + tm_w / 2 - 2.48, pill_y, 2.30, pill_h, fc=AMBER_FILL, ec=AMBER, lw=0.95, r=0.08)
    ax.text(
        tcx + tm_w / 2 - 2.48 + 1.15,
        pill_y + pill_h / 2,
        "Causal attention",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        color=AMBER,
    )

    arrow(ax, tcx, tm_y + tm_h, tcx, add1_y - 0.185)
    plus_node(ax, tcx, add1_y)

    # ---- Stage 2: Norm -> Router -> 2 of 64 Experts -> Weighted Sum -> Add ----
    split2_y = 3.06
    add2_y = 7.16
    arrow(ax, tcx, add1_y + 0.165, tcx, 3.26)
    ax.plot([tcx, res_x, res_x], [split2_y, split2_y, add2_y], color=MUTED, lw=0.95, zorder=2)
    arrow(ax, res_x, add2_y, tcx + 0.185, add2_y, lw=0.95)
    ax.scatter([tcx], [split2_y], s=14, color=MUTED, zorder=4)

    # RMSNorm 2
    n2_y, n2_h, n2_w = 3.28, 0.40, 2.75
    round_box(ax, tcx - n2_w / 2, n2_y, n2_w, n2_h, fc="white", ec=BORDER, lw=0.9, r=0.10)
    ax.text(tcx, n2_y + n2_h / 2, "RMSNorm", ha="center", va="center", fontsize=9.1, weight="bold")
    arrow(ax, tcx, n2_y + n2_h, tcx, 3.88)

    # Router
    rt_y, rt_h, rt_w = 3.90, 0.50, 3.10
    round_box(ax, tcx - rt_w / 2, rt_y, rt_w, rt_h, fc="white", ec=GREEN, lw=0.95, r=0.10)
    ax.text(tcx, rt_y + rt_h * 0.64, "Router", ha="center", va="center", fontsize=9.0, weight="bold", color=INK)
    ax.text(
        tcx,
        rt_y + rt_h * 0.27,
        f"sqrt-softplus · top-{top_k}",
        ha="center",
        va="center",
        fontsize=7.9,
        color=MUTED,
    )

    # 8 x 8 grid of 64 SwiGLU experts (2 highlighted)
    cell = 0.145
    cgap = 0.035
    grid_n = 8
    assert grid_n * grid_n == num_experts
    grid_side = grid_n * cell + (grid_n - 1) * cgap
    gx0 = tcx - grid_side / 2
    gy0 = 4.74

    # Crisp white backing card for the 8x8 expert pool
    round_box(
        ax,
        gx0 - 0.12,
        gy0 - 0.10,
        grid_side + 0.24,
        grid_side + 0.20,
        fc="white",
        ec="#d7dfe5",
        lw=0.8,
        r=0.09,
        zorder=2,
    )

    active_cells = {(2, 1), (5, 6)}  # (row_from_bottom, col) -> visually balanced
    active_centers: list[tuple[float, float]] = []
    for r in range(grid_n):
        for c in range(grid_n):
            cx_cell = gx0 + c * (cell + cgap)
            cy_cell = gy0 + r * (cell + cgap)
            is_active = (r, c) in active_cells
            round_box(
                ax,
                cx_cell,
                cy_cell,
                cell,
                cell,
                fc=GREEN if is_active else INACTIVE_FILL,
                ec="#006b4e" if is_active else INACTIVE_EDGE,
                lw=0.95 if is_active else 0.55,
                r=0.025,
                zorder=4 if is_active else 3,
            )
            if is_active:
                active_centers.append((cx_cell + cell / 2, cy_cell + cell / 2))

    # Side annotation next to the 64-expert grid
    ax.text(
        gx0 - 0.30,
        gy0 + grid_side * 0.62,
        "SwiGLU experts",
        ha="right",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=INK,
    )
    ax.text(
        gx0 - 0.30,
        gy0 + grid_side * 0.39,
        f"hidden width {expert_d_ff}",
        ha="right",
        va="center",
        fontsize=8.1,
        color=MUTED,
    )
    ax.text(
        gx0 - 0.30,
        gy0 + grid_side * 0.16,
        "no shared expert",
        ha="right",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )

    # Weighted sum node above the grid (clear gap above backing plate)
    ws_y = 6.56
    round_box(ax, tcx - 1.20, ws_y - 0.17, 2.40, 0.34, fc=GREEN_FILL, ec=GREEN, lw=0.9, r=0.10, zorder=4)
    ax.text(tcx, ws_y, "Weighted sum", ha="center", va="center", fontsize=8.8, weight="bold", color="#006b4e", zorder=5)

    # Sleek straight/near-straight routing paths: Router -> 2 active experts -> Weighted sum
    active_centers.sort(key=lambda pt: pt[0])
    for idx, (ex_x, ex_y) in enumerate(active_centers):
        rad_in = -0.08 if idx == 0 else 0.08
        rad_out = 0.08 if idx == 0 else -0.08
        arrow(
            ax,
            ex_x,
            rt_y + rt_h,
            ex_x,
            ex_y - cell / 2 - 0.015,
            color=GREEN,
            lw=1.35,
            head_len=3.3,
            head_wid=2.1,
            rad=rad_in,
            casing_color="white",
            zorder=5,
        )
        arrow(
            ax,
            ex_x,
            ex_y + cell / 2 + 0.015,
            ex_x,
            ws_y - 0.18,
            color=GREEN,
            lw=1.35,
            head_len=3.3,
            head_wid=2.1,
            rad=rad_out,
            casing_color="white",
            zorder=5,
        )

    # Weighted sum -> Add 2 -> Exit out top of Decoder layer
    arrow(ax, tcx, ws_y + 0.17, tcx, add2_y - 0.185)
    plus_node(ax, tcx, add2_y)
    arrow(ax, tcx, add2_y + 0.165, tcx, 7.82)

    save(fig, "architecture", release=True)


def draw_overview() -> None:
    """Wide, simplified diagram for the README: the layer stack, the tied embedding/head, and one
    inset with the pre-norm residual layer and top-k routing. Facts come from model.json."""
    m = json.loads((DATA / "model.json").read_text())
    pattern = list(m["layer_pattern"])
    repeats = int(m["n_layers"]) // len(pattern)
    num_experts, top_k = int(m["num_experts"]), int(m["top_k"])
    names = {"gdn": "GDN-2", "mha": "Attention"}

    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 12})
    fig = plt.figure(figsize=(9.0, 4.3))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 43)
    ax.axis("off")

    def box(x, w, y, h, text, *, fc="white", ec=BORDER, color=INK, weight="bold", size=12.5, lw=1.1):
        round_box(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=0.9)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, weight=weight,
                fontsize=size, zorder=5)

    def link(x0, y0, x1, y1, *, color=MUTED, ls="-", head=True, lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if head else "-",
                                     mutation_scale=11, color=color, lw=lw, ls=ls, zorder=3,
                                     shrinkA=0, shrinkB=0))

    # ---- main path
    y, h = 26.5, 6.0
    cy = y + h / 2
    ax.text(0.8, cy, "tokens", ha="left", va="center", color=MUTED, fontsize=11)
    x = 7.4
    link(5.6, cy, x, cy)
    box(x, 12.0, y, h, "Embedding", fc=SUBTLE_BG)
    emb_mid = x + 12.0 / 2
    x += 12.0 + 1.8
    frame_x0 = x
    link(x - 1.8, cy, x + 0.9, cy)
    x += 0.9
    for kind in pattern:
        w = 11.0 if kind == "mha" else 7.2
        fill, edge, ink = (AMBER_FILL, AMBER, AMBER) if kind == "mha" else (BLUE_FILL, BLUE, BLUE)
        box(x, w, y, h, names[kind], fc=fill, ec=edge, color=ink)
        x += w + 1.2
    frame_x1 = x - 0.3
    round_box(ax, frame_x0, y - 1.6, frame_x1 - frame_x0, h + 3.2, fc="none", ec=FRAME_EDGE, lw=1.0, r=1.2,
              zorder=1)
    ax.text(frame_x1 - 0.6, y + h + 2.4, f"\u00d7 {repeats}", ha="right", va="center", color=INK,
            weight="bold", fontsize=13)
    x = frame_x1 + 1.8
    link(frame_x1, cy, x, cy)
    box(x, 10.5, y, h, "RMSNorm", fc=SUBTLE_BG)
    x += 10.5 + 1.8
    link(x - 1.8, cy, x, cy)
    box(x, 13.4, y, h, "Output head", fc=SUBTLE_BG)
    head_mid = x + 13.4 / 2
    link(x + 13.4, cy, x + 15.0, cy)
    # tied embedding / output weights
    top = y + h
    ax.plot([emb_mid, emb_mid, head_mid, head_mid], [top, 39.4, 39.4, top], color=MUTED, lw=1.0,
            ls=(0, (3, 2.5)), zorder=1)
    ax.text((emb_mid + head_mid) / 2, 40.3, "embedding and output head share one weight matrix (tied)",
            ha="center", va="bottom", color=MUTED, fontsize=10.5)

    # ---- inset: one layer
    iy0, iy1 = 1.0, 20.0
    round_box(ax, 4.0, iy0, 82.0, iy1 - iy0, fc=SUBTLE_BG, ec=FRAME_EDGE, lw=1.0, r=1.2, zorder=0)
    ax.text(5.6, iy1 - 2.0, "Every layer", ha="left", va="center", color=INK, weight="bold", fontsize=12)
    ly, lh = 7.4, 5.0
    lc = ly + lh / 2
    x = 5.6
    link(x, lc, x + 2.4, lc)
    x += 2.4
    tap1 = x - 1.2
    box(x, 9.6, ly, lh, "RMSNorm", size=11.5)
    x += 9.6 + 1.8
    link(x - 1.8, lc, x, lc)
    box(x, 16.2, ly, lh, "GDN-2 or attention", size=11.5, weight="normal")
    x += 16.2 + 1.8
    link(x - 1.8, lc, x, lc)
    add1 = x + 1.0
    plus_node(ax, add1, lc, radius=1.0)
    x = add1 + 1.0 + 2.2
    link(add1 + 1.0, lc, x, lc)
    tap2 = add1 + 2.0
    box(x, 9.6, ly, lh, "RMSNorm", size=11.5)
    x += 9.6 + 1.8
    link(x - 1.8, lc, x, lc)
    # experts: one row of cells, top-k highlighted
    cw, gap = 0.30, 0.07
    ex0 = x
    chosen = {17, 44}
    for i in range(num_experts):
        on = i in chosen
        ax.add_patch(FancyBboxPatch((ex0 + i * (cw + gap), ly + 0.6), cw, lh - 1.2, boxstyle="square,pad=0",
                                    fc=GREEN if on else INACTIVE_FILL, ec=GREEN if on else INACTIVE_EDGE,
                                    lw=0.4, zorder=4))
    ex1 = ex0 + num_experts * (cw + gap) - gap
    ax.text((ex0 + ex1) / 2, ly + lh + 1.5, f"{num_experts} SwiGLU experts, {top_k} used per token",
            ha="center", va="center", color=GREEN, weight="bold", fontsize=11)
    # router: control path (dashed) to the selected experts
    rx, rw, ry, rh = (ex0 + ex1) / 2 - 5.5, 11.0, 1.9, 3.2
    ax.plot([x - 1.0, x - 1.0, rx], [lc, ry + rh / 2, ry + rh / 2], color=MUTED, lw=1.0, zorder=2)
    box(rx, rw, ry, rh, f"Router, top-{top_k}", ec=GREEN, color=GREEN, size=10.5)
    for i in sorted(chosen):
        cx = ex0 + i * (cw + gap) + cw / 2
        link(cx, ry + rh, cx, ly + 0.5, color=GREEN, ls=(0, (2, 1.6)), lw=1.0)
    x = ex1 + 1.8
    link(ex1 + 0.2, lc, x, lc)
    add2 = x + 1.0
    plus_node(ax, add2, lc, radius=1.0)
    link(add2 + 1.0, lc, add2 + 3.4, lc)
    # residual skips over each branch
    for t, a in ((tap1, add1), (tap2, add2)):
        ax.plot([t, t, a, a], [lc, lc + 5.6, lc + 5.6, lc + 1.0], color=MUTED, lw=1.0, zorder=1)

    save(fig, "architecture_overview", release=True)


if __name__ == "__main__":
    draw_architecture()
    draw_overview()
