#!/usr/bin/env python3
"""Draw the zero-shot comparison figures and write results/benchmarks.md.

Reads only results/data/benchmarks.json; writes results/figures/bench_vs_compute.{svg,png},
bench_vs_compute-dark.png, bench_vs_params.{svg,png}, bench_vs_params-dark.png and
results/benchmarks.md. Prints the two README tables to stdout. No network.

    python results/make_benchmarks.py      # needs matplotlib and rsvg-convert
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

import make_figures
from make_figures import save

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "benchmarks.json").read_text())
MODELS = DATA["models"]

# (task, metric, column label, chance %); the first nine form the README average.
CORE = [
    ("blimp", "acc", "BLiMP", 50), ("lambada_openai", "acc", "LAMBADA", 0),
    ("hellaswag", "acc_norm", "HellaSwag", 25), ("piqa", "acc_norm", "PIQA", 50),
    ("winogrande", "acc", "WinoGrande", 50), ("arc_easy", "acc_norm", "ARC-E", 25),
    ("arc_challenge", "acc_norm", "ARC-C", 25), ("openbookqa", "acc_norm", "OBQA", 25),
    ("sciq", "acc_norm", "SciQ", 25),
]
MORE = [
    ("social_iqa", "acc", "SocialIQA", 33), ("copa", "acc", "COPA", 50),
    ("swag", "acc_norm", "SWAG", 25), ("race", "acc", "RACE", 25),
    ("commonsense_qa", "acc", "CSQA", 20), ("boolq", "acc", "BoolQ", 62),
    ("wsc273", "acc", "WSC273", 50), ("truthfulqa_mc2", "acc", "TQA MC2", None),
]
CHANCE_CORE = sum(c for *_, c in CORE) / len(CORE)

INK, MUTED = make_figures.INK, make_figures.MUTED
ACCENT = "#0072b2"
BASE_DOT, BASE_TEXT, CHANCE = "#8f9ca6", "#5d6b76", "#c3cdd4"
make_figures.DARK_COLORS.update({BASE_DOT: "#7f91a0", BASE_TEXT: "#a9b8c5"})


def score(m: dict, task: str, metric: str) -> float:
    return 100 * float(m["results"][task][f"{metric},none"])


def core_mean(m: dict) -> float:
    return sum(score(m, t, k) for t, k, *_ in CORE) / len(CORE)


def all_mean(m: dict) -> float:
    cols = CORE + MORE
    return sum(score(m, t, k) for t, k, *_ in cols) / len(cols)


def sci(x: float) -> str:
    exp = len(str(int(x))) - 1
    return f"{x / 10 ** exp:.1f}×10<sup>{exp}</sup>"


def short(n: int) -> str:
    for div, unit in ((10 ** 12, "T"), (10 ** 9, "B"), (10 ** 6, "M")):
        if n >= div:
            v = n / div
            if v < 10 and unit == "M":
                return f"{v:.2f}{unit}"
            return f"{v:.1f}{unit}" if v < 10 and unit == "B" else f"{v:.0f}{unit}"
    return str(n)


# ------------------------------------------------------------------ figures
# Label positions in data coordinates (x, y, alignment, leader line), per figure, chosen so that
# nothing overlaps at 720 px; a leader line joins a label placed away from its point.
LABELS = {
    "compute": {
        "Kefir 1 Base": (2.6e18, 48.9, "left", False), "Kefir 1": (2.6e18, 46.1, "left", False),
        "GPT-2 124M FineWeb-Edu": (3e19, 45.6, "left", True),
        "TinyBrainBot 61M-A10M": (4e18, 42.0, "left", True), "Cerebras-GPT-111M": (4e18, 38.9, "left", True),
        "Pythia-70M, step 3,000": (4e18, 36.1, "left", True),
        "SmolLM2-135M": (1.7e21, 53.2, "left", False), "Qwen3-0.6B-Base": (6.5e22, 60.9, "right", False),
    },
    "params": {
        "Kefir 1 Base": (7.0e6, 48.6, "right", False), "Kefir 1": (7.0e6, 46.2, "right", False),
        "TinyBrainBot 61M-A10M": (1.15e7, 42.9, "left", False), "Pythia-70M, step 3,000": (2.1e7, 37.0, "left", False),
        "Cerebras-GPT-111M": (8.2e7, 39.9, "left", False), "GPT-2 124M FineWeb-Edu": (9.8e7, 46.4, "left", False),
        "SmolLM2-135M": (1.25e8, 53.4, "left", False), "Qwen3-0.6B-Base": (3.8e8, 60.9, "right", False),
    },
}


def scatter(kind: str) -> None:
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 12})
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    fig.subplots_adjust(left=.125, right=.975, top=.965, bottom=.17)
    xkey = "training_flops_estimate" if kind == "compute" else "active_nonembedding_params"
    ax.axhline(CHANCE_CORE, color=CHANCE, lw=1, ls=(0, (4, 3)), zorder=1)
    for m in sorted(MODELS, key=lambda m: m["kefir"]):
        x, y = m[xkey], core_mean(m)
        if m["kefir"]:
            chat = "Chat" in m["label"]
            ax.scatter([x], [y], s=70, zorder=4, color="white" if chat else ACCENT,
                       edgecolor=ACCENT, linewidth=1.8)
            style = dict(color=ACCENT, weight="bold", fontsize=12.5)
        else:
            ax.scatter([x], [y], s=34, zorder=3, color=BASE_DOT, linewidth=0)
            style = dict(color=BASE_TEXT, fontsize=11)
        tx, ty, ha, leader = LABELS[kind][m["label"]]
        arrow = dict(arrowstyle="-", color=CHANCE, lw=.8, shrinkA=6, shrinkB=4) if leader else None
        ax.annotate(m["label"], (x, y), xytext=(tx, ty), textcoords="data", ha=ha, va="center",
                    arrowprops=arrow, **style)
    ax.set_xscale("log")
    if kind == "compute":
        ax.set_xlim(1.5e17, 4e23)
        ax.set_xlabel("Estimated training compute (FLOPs, log scale)")
        ticks = [1e18, 1e19, 1e20, 1e21, 1e22, 1e23]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{round(math.log10(v))}}}$"))
    else:
        ax.set_xlim(1.5e6, 1.6e9)
        ax.set_xlabel("Active non-embedding parameters per token (log scale)")
        ticks = [3e6, 1e7, 3e7, 1e8, 3e8, 1e9]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f"{v / 1e9:.0f}B" if v >= 1e9 else f"{v / 1e6:.0f}M"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(28, 64)
    ax.set_ylabel("Mean accuracy, 9 tasks (%)")
    ax.text(ax.get_xlim()[1], CHANCE_CORE + .6, f"chance {CHANCE_CORE:.1f}", ha="right", va="bottom",
            fontsize=10, color=MUTED)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)
    save(fig, f"bench_vs_{kind}", release=True)


# Label placement for the compute figure: (offset in points, horizontal alignment) per model.
HERO_LABELS = {
    "Qwen3-0.6B-Base": ((-9, 9), "right"), "SmolLM2-135M": ((9, 0), "left"),
    "GPT-2 124M FineWeb-Edu": ((9, 0), "left"), "TinyBrainBot 61M-A10M": ((9, 0), "left"),
    "Cerebras-GPT-111M": ((9, 4), "left"), "Pythia-70M, step 3,000": ((9, -5), "left"),
}


def compute_figure() -> None:
    """Mean accuracy against estimated pretraining compute relative to Kefir 1, labels beside the points.

    Both Kefir checkpoints are drawn, as in the parameter figure: Kefir 1 (chat) filled, Kefir 1 Base hollow."""
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11.5,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 11.5})
    kefir = next(m for m in MODELS if m["label"] == "Kefir 1 Base")
    k_flops = kefir["training_flops_estimate"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fig.subplots_adjust(left=.105, right=.975, top=.97, bottom=.15)
    ax.axhline(CHANCE_CORE, color=CHANCE, lw=1, ls=(0, (4, 3)), zorder=1)
    for m in MODELS:
        x, y = m["training_flops_estimate"] / k_flops, core_mean(m)
        if m["kefir"]:
            continue
        ax.scatter([x], [y], s=38, zorder=3, color=BASE_DOT, linewidth=0)
        off, ha = HERO_LABELS[m["label"]]
        ax.annotate(f"{m['label']}  {y:.1f}", (x, y), xytext=off, textcoords="offset points", ha=ha,
                    va="center", color=BASE_TEXT, fontsize=10.5)
    chat = next(m for m in MODELS if m["kefir"] and m is not kefir)
    yb, yc = core_mean(kefir), core_mean(chat)
    xc = chat["training_flops_estimate"] / k_flops
    kefir_points(ax, (xc, yc), (1, yb))
    # Both labels sit above the pair, in the order of the points (Base scores higher); below them the
    # TinyBrainBot label would collide.
    ax.annotate(f"Kefir 1 Base  {yb:.1f}", (1, yb), xytext=(0, 31), textcoords="offset points", ha="center",
                va="center", color=ACCENT, fontsize=10.5)
    ax.annotate(f"Kefir 1  {yc:.1f}", (xc, yc), xytext=(0, 17), textcoords="offset points",
                ha="center", va="center", color=ACCENT, weight="bold", fontsize=11.5)
    ax.set_xscale("log")
    ax.set_xlim(.1, 3e5)
    ticks = [.1, 1, 10, 100, 1e3, 1e4, 1e5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×" if v < 1e3 else f"{v:,.0f}×"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Estimated pretraining compute relative to Kefir 1 (log scale)")
    ax.set_ylim(28, 63)
    ax.set_ylabel("Mean accuracy, 9 tasks (%)")
    ax.text(ax.get_xlim()[1], CHANCE_CORE + .6, f"chance {CHANCE_CORE:.1f}", ha="right", va="bottom",
            fontsize=10, color=MUTED)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "bench_vs_compute", release=True)


def kefir_points(ax, chat_xy, base_xy) -> None:
    """Kefir 1 (chat) as a filled point, Kefir 1 Base as a hollow ring drawn over it."""
    ax.scatter([chat_xy[0]], [chat_xy[1]], s=70, zorder=5, color=ACCENT, linewidth=0)
    ax.scatter([base_xy[0]], [base_xy[1]], s=120, zorder=6, facecolor="none", edgecolor=ACCENT,
               linewidth=1.6)


PARAM_LABELS = {  # offset in points and alignment, per model
    "Qwen3-0.6B-Base": ((-10, -10), "right"), "SmolLM2-135M": ((10, 0), "left"),
    "GPT-2 124M FineWeb-Edu": ((10, 0), "left"), "TinyBrainBot 61M-A10M": ((10, 0), "left"),
    "Cerebras-GPT-111M": ((10, 0), "left"), "Pythia-70M, step 3,000": ((10, -2), "left"),
}


def params_figure() -> None:
    """README figure: mean accuracy over nine tasks against active non-embedding parameters."""
    plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 11.5,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.labelsize": 11})
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    fig.subplots_adjust(left=.095, right=.975, top=.97, bottom=.15)
    ax.grid(axis="y", color=make_figures.GRID, lw=.7)
    ax.grid(axis="x", visible=False)
    kefir = next(m for m in MODELS if m["kefir"] and m["label"] != "Kefir 1 Base")
    base = next(m for m in MODELS if m["label"] == "Kefir 1 Base")
    for m in MODELS:
        if m["kefir"]:
            continue
        x, y = m["active_nonembedding_params"], core_mean(m)
        ax.scatter([x], [y], s=44, color=BASE_DOT, zorder=3, linewidth=0)
        off, ha = PARAM_LABELS[m["label"]]
        ax.annotate(f"{m['label']}  ", (x, y), xytext=off, textcoords="offset points", ha=ha, va="center",
                    color=BASE_TEXT, fontsize=10.5)
    x, yc = kefir["active_nonembedding_params"], core_mean(kefir)
    xb, yb = base["active_nonembedding_params"], core_mean(base)
    kefir_points(ax, (x, yc), (xb, yb))
    ax.annotate("Kefir 1 Base", (xb, yb), xytext=(11, 8), textcoords="offset points", ha="left", va="center",
                color=ACCENT, fontsize=11)
    ax.annotate("Kefir 1", (x, yc), xytext=(11, -8), textcoords="offset points", ha="left", va="center",
                color=ACCENT, weight="bold", fontsize=12.5)
    ax.set_xscale("log")
    ax.set_xlim(4.5e6, 1.1e9)
    ax.xaxis.set_major_locator(FixedLocator([1e7, 3e7, 1e8, 3e8, 1e9]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "1B" if v >= 1e9 else f"{v / 1e6:.0f}M"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Active parameters per token, embeddings excluded (log scale)")
    ax.set_ylim(34, 62.5)
    ax.yaxis.set_major_locator(FixedLocator([35, 40, 45, 50, 55, 60]))
    ax.set_ylabel("Mean zero-shot accuracy, 9 tasks (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "bench_vs_params", release=True)


# ------------------------------------------------------------------ tables
def table(cols, with_mean: bool) -> str:
    head = ["Model"] + [label for _, _, label, _ in cols] + (["Mean"] if with_mean else [])
    rows = []
    for m in MODELS:
        vals = [score(m, t, k) for t, k, *_ in cols] + ([core_mean(m)] if with_mean else [])
        rows.append((m["label"], [round(v, 1) for v in vals]))
    best = [max(r[1][i] for r in rows) for i in range(len(head) - 1)]
    out = ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    for label, vals in rows:
        cells = [f"**{v:.1f}**" if v == b else f"{v:.1f}" for v, b in zip(vals, best)]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def metric_line(cols) -> str:
    return ", ".join(f"{label} `{k}`" for _, k, label, _ in cols)


def models_table() -> str:
    out = ["| Model | Checkpoint | Active non-embedding parameters | Training tokens | Estimated compute (FLOPs) |",
           "|---|---|---:|---:|---:|"]
    for m in MODELS:
        tok = ("~" if m.get("approx_tokens") else "") + short(m["training_tokens"])
        out.append(f"| {m['label']} | `{m['checkpoint']}` | {short(m['active_nonembedding_params'])} | {tok} | "
                   f"{('~' if m.get('approx_tokens') else '')}{sci(m['training_flops_estimate'])} |")
    return "\n".join(out)


def extra_table() -> str:
    out = ["| Model | TruthfulQA MC1 `acc` | WikiText bits/byte ↓ | WikiText word perplexity ↓ | Mean, 9 tasks | Mean, 17 tasks |",
           "|---|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        w = m["results"].get("wikitext")
        bpb = f"{w['bits_per_byte,none']:.3f}" if w else "–"
        wp = f"{w['word_perplexity,none']:.1f}" if w else "–"
        out.append(f"| {m['label']} | {score(m, 'truthfulqa_mc1', 'acc'):.1f} | {bpb} | {wp} | "
                   f"{core_mean(m):.1f} | {all_mean(m):.1f} |")
    return "\n".join(out)


def sources_table() -> str:
    out = ["| Model | Parameter count source | Token count source |", "|---|---|---|"]
    for m in MODELS:
        out.append(f"| {m['label']} | {m['params_source']} | {m['tokens_source']} |")
    return "\n".join(out)


def write_markdown() -> None:
    h = DATA["harness"]
    n = {t: len(MODELS) for t, *_ in CORE + MORE}
    text = f"""# Zero-shot benchmarks

Kefir 1 Base, Kefir 1 and six small open base models, all evaluated by us with the same harness and
settings. This page is written by [`make_benchmarks.py`](make_benchmarks.py) from
[`data/benchmarks.json`](data/benchmarks.json), which holds every score, including the 67 BLiMP paradigms.

## Settings

- {h['package']}, {h['num_fewshot']}-shot, {h['precision']}, {h['hardware']}, batch size {h['batch_size']}, no sample
  limit, `random_seed={h['random_seed']}`, `numpy_random_seed={h['numpy_random_seed']}`,
  `torch_random_seed={h['torch_random_seed']}`.
- Baselines: {h['baselines']}.
- Kefir 1 and Kefir 1 Base: {h['kefir']}.
- Kefir's SuperBPE tokenizer can merge across the prompt/answer boundary; the adapter encodes them separately; 0.65% of
  requests in 5 tasks were affected and rerun.
- No standard errors were computed (`bootstrap_iters=0`). Qwen3-0.6B-Base WikiText ran out of T4 memory and is
  missing.
- Every model scored the same items; sample counts are those of the harness tasks (for example HellaSwag 10,042,
  ARC-Easy 2,376, BLiMP 67,000).
- Kefir 1 is scored as raw text, without its chat template, like the base models.
- Overlap between these benchmarks and the Kefir 1 pretraining corpus was not checked.
- To rerun one model: `PYTHONPATH=src python results/bench_lm.py kefir:path/to/model.safetensors out.json`
  (or `hf:repo_id[@revision]` for a baseline). It needs `lm_eval[hf]==0.4.13`. The script is
  [`bench_lm.py`](bench_lm.py). Without `--tasks` it runs the nine core tasks; the 17 tasks of the tables below,
  plus TruthfulQA MC1 and WikiText, are
  `--tasks {",".join(t for t, *_ in CORE + MORE)},truthfulqa_mc1,wikitext`.
- TinyBrainBot ships its own model code, so its spec needs the `+trc` suffix
  (`hf:nkthebass/tinybrainbot-61m-a10m-v4.1-moe-base+trc`), which passes `trust_remote_code=True` and runs
  third-party code from that repository. The revision used was not recorded; review the code at the revision you
  pin (`hf:repo@revision+trc`) before running it.

Scores are percentages; higher is better except where marked ↓. The best value in each column is in bold.
Metrics: {metric_line(CORE + MORE)}.

## Models

Compute is estimated as **6 × active non-embedding parameters × training tokens**. It ignores attention and
recurrent-state FLOPs, tokenizer differences (a token is not the same amount of text across models) and hardware
efficiency; it is a coarse scale, not a measurement. For Kefir 1 and Kefir 1 Base, the active count excludes the tied 8,192 × 256
embedding/output matrix (9,987,992 − 2,097,152 = 7,890,840) and the tokens are the 40,632,320,000 target
positions seen in pretraining, end-of-document padding included.

{models_table()}

{sources_table()}

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/bench_vs_compute-dark.png" width="720">
  <img src="figures/bench_vs_compute.png" width="720" alt="Mean zero-shot accuracy over nine tasks against estimated pretraining compute relative to Kefir 1 for the eight models; Kefir 1 and Kefir 1 Base share the same compute.">
</picture>

*Mean zero-shot accuracy over the nine core tasks against estimated pretraining compute relative to Kefir 1
(log scale; Kefir 1 ≈ 1.9 × 10¹⁸ FLOPs). Kefir 1 (filled) and Kefir 1 Base (hollow) share the same pretraining
compute. Dashed line: mean chance level of the nine tasks (LAMBADA, an open-ended completion, counts as 0 %).*

## Nine core tasks

{table(CORE, True)}

## Eight further tasks

{table(MORE, False)}

## Other measurements

{extra_table()}

The 17-task mean averages the two tables above. Below Qwen3, every model stays near chance on CommonsenseQA (20 %)
and at or below the majority class on BoolQ (62 %). WikiText bits per byte does not depend on the tokenizer.
"""
    (HERE / "benchmarks.md").write_text(text)


def main() -> None:
    compute_figure()
    params_figure()
    write_markdown()
    print(models_table(), "\n")
    print(table(CORE, True), "\n")
    print(table(MORE, False))


if __name__ == "__main__":
    main()
