#!/usr/bin/env python3
"""Draw the nine measurement figures from the aggregated tables in results/data.

The architecture diagram is owned by results/make_architecture.py.

Reads only results/data/*.csv|json; writes results/figures/<name>.svg and <name>.png.
Release plots also get a dark PNG via rsvg-convert.
Deterministic for a fixed matplotlib version: fixed SVG hash salt, no timestamps in metadata.

    python results/make_figures.py
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figures"

INK, MUTED, GRID = "#182b3a", "#52616e", "#e5ebef"
plt.rcParams.update({
    "svg.hashsalt": "kefir-results", "svg.fonttype": "path",
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#a9b5bd",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .65, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "figure.dpi": 100, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 3, "ytick.major.size": 3,
})


def heading(fig, title, subtitle):
    fig.text(.09, .955, title, fontsize=15, weight="bold", va="top")
    fig.text(.09, .895, subtitle, fontsize=10, color=MUTED, va="top")


def clean(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", visible=False)


# Okabe-Ito palette (colour-blind safe)
BLUE, ORANGE, GREEN, PURPLE, SKY, VERMIL, YELLOW, GREY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#7f7f7f"
TRAJ_STYLE = {
    "flat-b32-s17": dict(color=BLUE, ls="-", label="flat, batch 32, seed 17"),
    "flat-b64-s17": dict(color=ORANGE, ls="-", label="flat, batch 64, seed 17"),
    "flat-b64-s18": dict(color=ORANGE, ls="--", label="flat, batch 64, seed 18"),
    "latedecay-b32-s17": dict(color=SKY, ls="-", label="late decay from 10k, batch 32, seed 17"),
    "latedecay-b64-s17": dict(color=VERMIL, ls="-", label="late decay from 10k, batch 64, seed 17"),
    "wd-b32-s17": dict(color=PURPLE, ls="-", label="warm-up + decay, seed 17"),
    "wd-b32-s18": dict(color=PURPLE, ls="--", label="warm-up + decay, seed 18"),
    "wsd-b32-s17": dict(color=GREEN, ls="-", label="warm-up + stable + decay, seed 17"),
    "wsd-b32-s18": dict(color=GREEN, ls="--", label="warm-up + stable + decay, seed 18"),
}


# ------------------------------------------------------------------ data access (read-only)
def read_csv(name: str) -> list[dict]:
    with open(DATA / name, newline="") as fh:
        return list(csv.DictReader(fh))


# Palette substitutions in Matplotlib's generated SVG preserve geometry and opacity.
# Only figures embedded on release pages get dark variants.
DARK_COLORS = {
    INK: "#e6edf3", MUTED: "#b4c2cf", GRID: "#2e3e4e", "#a9b5bd": "#697e91",
    "#c3cdd4": "#4d6377", "#f7f9fb": "#172534", "#ffffff": "#202e3e",
    "#edf2f6": "#26384a", "#c8d2da": "#4a6175", "#d7dfe5": "#506578",
    "#0072b2": "#70bff4", "#eaf3fa": "#193c56", "#d55e00": "#ffa46d",
    "#fdf0e8": "#4a2f26", "#009e73": "#58d5ac", "#e6f6f1": "#193d35",
    "#006b4e": "#70dfbb",
    "#a05b00": "#ffbf69", "#963c98": "#e9a3e9", "#a72c4e": "#ff8eac",
    "#166c75": "#6fd8de", "#527208": "#b3db73", "#5a53a4": "#a9abfc",
    "#78603b": "#d9bfa2", "#a84c16": "#f6a582", "#356b53": "#89cbb1",
    "#586b8c": "#a8bfdf", "#70646d": "#cbb9ca",
}


def save(fig, name: str, *, release: bool = False) -> None:
    FIGS.mkdir(exist_ok=True)
    svg = FIGS / f"{name}.svg"
    fig.savefig(svg, transparent=release, metadata={"Date": None, "Creator": None})
    source = "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    fig.savefig(FIGS / f"{name}.png", transparent=release, metadata={"Software": None})
    if release:
        rules = []
        for channel in ("fill", "stroke"):
            for light_color, dark_color in DARK_COLORS.items():
                declaration = f"{channel}: {light_color}"
                if declaration in source:
                    rules.append(f'[style*="{declaration}"]{{{channel}:{dark_color}!important}}')
        style = '<style>@media (prefers-color-scheme: dark) {' + "".join(rules) + "}</style>"
        svg.write_text(source.replace("<defs>", style + "\n <defs>", 1))
        dark = re.sub(r"(?<=: )#[0-9a-f]{6}\b",
                      lambda color: DARK_COLORS.get(color[0], color[0]), source)
        width = round(fig.get_figwidth() * 300)
        subprocess.run(
            ["rsvg-convert", "--width", str(width), "--output", str(FIGS / f"{name}-dark.png")],
            input=dark.encode(),
            check=True,
        )
    else:
        svg.write_text(source)
    plt.close(fig)


# ------------------------------------------------------------------ pretraining
def fig_pretrain_loss() -> None:
    tr, va = read_csv("pretrain_train_binned.csv"), read_csv("pretrain_validation.csv")
    fork = next(float(r["targets_seen"]) for r in va if r["step"] == "280000") / 1e9
    valid = [r for r in va if r["val_loss"]]
    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    fig.subplots_adjust(left=.12, right=.97, top=.94, bottom=.19)
    x = [float(r["targets_seen"]) / 1e9 for r in tr]
    ax.fill_between(x, [float(r["train_loss_p10"]) for r in tr],
                    [float(r["train_loss_p90"]) for r in tr], color=BLUE, alpha=.15, lw=0)
    ax.plot(x, [float(r["train_loss_mean"]) for r in tr], color=BLUE, label="Training mean")
    ax.plot([float(r["targets_seen"]) / 1e9 for r in valid], [float(r["val_loss"]) for r in valid],
            color=VERMIL, marker="o", ms=3, lw=1.4, label="Held-out validation")
    ax.axvline(fork, color=MUTED, ls=":", lw=1.1, label="Cooldown starts")
    ax.set(xscale="log", xlabel="Training targets (billions, log scale)", ylabel="Loss (nats / target)")
    ax.set_xticks([.1, 1, 10, 40], ["0.1", "1", "10", "40"])
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(.91, 1), fontsize=10.5)
    clean(ax)
    save(fig, "pretrain_loss", release=True)


def fig_pretrain_lr() -> None:
    tr = read_csv("pretrain_train_binned.csv")
    s = json.loads((DATA / "pretrain_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    heading(fig, "Pretraining learning rate", "Linear warm-up → constant rate → cosine cooldown")
    fig.subplots_adjust(left=.10, right=.97, top=.75, bottom=.19)
    ax.plot([0]+[int(r["step_last"])/1000 for r in tr], [0]+[float(r["lr_at_last_step"])*1e4 for r in tr], color=BLUE)
    ax.axvline(280, color=MUTED, ls=":", lw=1)
    ax.set(xlim=(0, 320), ylim=(0, 3.5), xlabel="Optimizer updates (thousands)", ylabel="Learning rate (×10⁻⁴)")
    ax.set_xticks([0, 50, 100, 150, 200, 250, 310])
    ax.text(130, 3.12, f"Peak {s['lr_peak']:g}", ha="center", fontsize=10, color=BLUE)
    ax.annotate(f"Warm-up ends at {s['lr_first_step_at_peak']:,}", (s['lr_first_step_at_peak']/1000, 3),
                xytext=(55, 1.6), fontsize=10, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.8))
    ax.annotate(f"Final {s['lr_final']:.0e}", (310, .3), xytext=(235, .75), color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=.8))
    clean(ax)
    save(fig, "pretrain_lr")


def theme_label(t):
    return {"general_prose": "General prose", "physical_world": "Physical world",
            "practical_life": "Practical life", "spatial_reasoning": "Spatial reasoning",
            "verbal_reasoning": "Verbal reasoning"}.get(t, t.replace("_", " & ").capitalize())


def fig_pretrain_themes() -> None:
    va, te = read_csv("pretrain_validation.csv"), read_csv("pretrain_themes_endpoints.csv")
    rows = [r for r in va if r["thematic_protocol"] == "bf16_logged_during_training" and int(r["step"]) >= 10000]
    endpoint = next(r for r in va if r["step"] == "310000")
    fork = next(float(r["targets_seen"]) / 1e9 for r in va if r["step"] == "280000")
    series = sorted(te, key=lambda row: float(row["loss_310000"]), reverse=True)
    colors = (BLUE, "#a05b00", "#963c98", "#a72c4e", "#166c75", "#527208",
              "#5a53a4", "#78603b", "#a84c16", "#356b53", "#586b8c", "#70646d")
    fig, ax = plt.subplots(figsize=(6.4, 3.45))
    fig.subplots_adjust(left=.12, right=.62, top=.95, bottom=.18)
    x = [float(row["targets_seen"]) / 1e9 for row in rows]
    styles = ("-", "--", "-.", ":")
    for index, (row, color) in enumerate(zip(series, colors, strict=True)):
        theme = row["theme"]
        ax.plot(x, [float(v[theme]) for v in rows], color=color,
                ls=styles[index % len(styles)], lw=1.3, label=theme_label(theme))
        ax.plot(float(endpoint["targets_seen"]) / 1e9, float(row["loss_310000"]), "^", color=color, ms=4)
    ax.axvline(fork, color=MUTED, ls=":", lw=.8)
    ax.set(xlim=(0, 42), xticks=[0, 10, 20, 30, 40], ylabel="Loss (nats / target)",
           xlabel="Training targets (billions)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), frameon=False,
              fontsize=10, handlelength=1.8, labelspacing=.45)
    clean(ax)
    save(fig, "pretrain_themes", release=True)


# ------------------------------------------------------------------ SFT
def sft_tables():
    traj = {r["trajectory"]: r for r in read_csv("sft_trajectories.csv")}
    upd = defaultdict(list)
    for r in read_csv("sft_updates.csv"):
        upd[r["segment"]].append(r)
    ev = defaultdict(list)
    for r in read_csv("sft_evaluations.csv"):
        ev[r["segment"]].append(r)
    return traj, upd, ev


def traj_updates(traj, upd, tid):
    out = []
    for seg in traj[tid]["segments"].split(";"):
        out.extend(upd[seg])
    return out


def traj_evals(traj, ev, tid):
    out = list(ev["parent-310000"])
    for seg in traj[tid]["segments"].split(";"):
        out.extend(ev[seg])
    return out


def binned(updates, width=500):
    acc = defaultdict(lambda: [0.0, 0.0])
    for r in updates:
        k = (int(r["chats_seen"]) - 1) // width
        w = float(r["update_targets"])
        acc[k][0] += w * float(r["train_loss"])
        acc[k][1] += w
    ks = sorted(acc)
    return [(k + 1) * width for k in ks], [acc[k][0] / acc[k][1] for k in ks]


PANELS = [
    ("a  Batch size", ["flat-b32-s17", "flat-b64-s17", "flat-b64-s18"], (0, 15500)),
    ("b  Schedule · batch 32", ["flat-b32-s17", "wd-b32-s17", "wd-b32-s18", "wsd-b32-s17", "wsd-b32-s18"], (0, 15500)),
    ("c  Late decay", ["flat-b32-s17", "latedecay-b32-s17", "flat-b64-s17", "latedecay-b64-s17"], (9500, 15500)),
]
SHORT = {
    "flat-b32-s17": "Constant · B32 · S17", "flat-b64-s17": "Constant · B64 · S17",
    "flat-b64-s18": "Constant · B64 · S18", "latedecay-b32-s17": "Late decay · B32 · S17",
    "latedecay-b64-s17": "Late decay · B64 · S17", "wd-b32-s17": "Warm-up + decay · S17",
    "wd-b32-s18": "Warm-up + decay · S18", "wsd-b32-s17": "Warm-up + stable + decay · S17",
    "wsd-b32-s18": "Warm-up + stable + decay · S18",
}


def kchats(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}"))
    ax.set_xlabel("Conversations (thousands)")


def sft_panels(training=False):
    traj, upd, ev = sft_tables()
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 5.8), sharey=True)
    heading(fig, "SFT training loss" if training else "SFT validation loss",
            "Target-weighted means in 500-conversation bins" if training else "512 held-out conversations · assistant-token NLL · lower is better")
    fig.subplots_adjust(left=.09, right=.97, top=.74, bottom=.40, wspace=.19)
    for ax, (title, ids, xl) in zip(axes, PANELS):
        for tid in ids:
            if training:
                x,y=binned(traj_updates(traj,upd,tid))
            else:
                rows=traj_evals(traj,ev,tid)
                x,y=[int(r['chats_seen']) for r in rows], [float(r['val_nll']) for r in rows]
            points=[(xx,yy) for xx,yy in zip(x,y) if xx>=xl[0]]
            st=TRAJ_STYLE[tid]
            # Short local legends avoid a dense, detached nine-series key.
            label=SHORT[tid].replace(" · S17", "" if title.startswith('c') else " · S17")
            label=label.replace("Warm-up + stable + decay", "Warm-up / stable / decay")
            ax.plot(*zip(*points), color=st['color'], ls=st['ls'], marker=None if training else 'o', ms=3, label=label)
        ax.set_xlim(*xl)
        ax.set_title(title, pad=12)
        kchats(ax)
        clean(ax)
        ax.set_xticks([10000,12500,15000] if xl[0]>0 else [0,5000,10000,15000])
        ax.legend(loc='upper left', bbox_to_anchor=(-.04,-.35), frameon=False, fontsize=8.5, handlelength=1.7, labelspacing=.6)
    axes[0].set_ylabel("Assistant NLL (nats / target)")
    if not training:
        parent=float(ev['parent-310000'][0]['val_nll'])
        axes[0].set_ylim(1.75,2.16)
        for ax in axes:
            ax.axhline(parent, color=MUTED, ls=':', lw=.8)
        # Highlight both unrepeated evaluations from the actual rows.
        for ax,tid,chats in [(axes[1],'wd-b32-s17',15000),(axes[2],'latedecay-b32-s17',14000)]:
            row=next(r for r in traj_evals(traj,ev,tid) if int(r['chats_seen'])==chats)
            ax.plot(chats,float(row['val_nll']), 'o', ms=8, mfc='none', mec=INK, mew=.8)
        note="Dotted: pretrained parent. Circled: two unrepeated evaluations. B = batch; S = data-order seed."
    else:
        note="B = batch size; S = data-order seed. One run per line; batch composition varies across bins."
    fig.text(.09,.035,note,fontsize=9,color=MUTED)
    save(fig, "sft_train_loss" if training else "sft_validation_nll")


def fig_sft_validation():
    sft_panels()


def fig_sft_train():
    sft_panels(training=True)


def fig_sft_lr() -> None:
    traj, upd, _ = sft_tables()
    fig, axes = plt.subplots(1,4,figsize=(8.8,3.6),sharex=True,sharey=True)
    heading(fig,"SFT learning-rate schedules","Batch 32 · data-order seed 17 · recorded optimizer rates")
    fig.subplots_adjust(left=.09,right=.97,top=.67,bottom=.24,wspace=.18)
    for ax,tid,title in zip(axes,["flat-b32-s17","latedecay-b32-s17","wd-b32-s17","wsd-b32-s17"],
                            ["Constant","Late decay","Warm-up + decay","Warm-up / stable\n/ decay"]):
        rows=traj_updates(traj,upd,tid)
        ax.plot([int(r['chats_seen'])/1000 for r in rows], [float(r['lr'])*1e5 for r in rows],color=TRAJ_STYLE[tid]['color'])
        ax.set(xlim=(0,15),ylim=(0,3.3),xticks=[0,5,10,15])
        ax.set_title(title,fontsize=10,pad=10)
        clean(ax)
    axes[0].set_ylabel("Learning rate (×10⁻⁵)")
    fig.supxlabel("Training conversations (thousands)",y=.065,fontsize=10)
    save(fig,"sft_lr_schedules")


METRICS = [
    ("blimp_of_64", "BLiMP · of 64", (0,64)),
    ("arc_raw_of_30", "ARC-Easy · of 30", (0,30)),
    ("instr_constraints_of_36", "Instructions · of 36", (0,36)),
    ("stopping_native_of_16", "Stopping · of 16", (0,16)),
    ("open80_native_end_of_80", "Open80 stopping · of 80", (0,80)),
    ("retention_bpb", "Raw-fact bits / byte ↓", (1.1,1.45)),
]


def fig_sft_benchmarks() -> None:
    traj, _, ev=sft_tables()
    fig, axes=plt.subplots(2,3,figsize=(8.8,6.7))
    heading(fig,"SFT benchmark trajectories","Development split · constant learning rate · FP16 in-loop evaluation")
    fig.subplots_adjust(left=.09,right=.97,top=.76,bottom=.22,hspace=.58,wspace=.28)
    for ax,(col,title,limits) in zip(axes.flat,METRICS):
        for tid in ['flat-b32-s17','flat-b64-s17','flat-b64-s18']:
            rows=[r for r in traj_evals(traj,ev,tid) if r[col]!='']
            st=TRAJ_STYLE[tid]
            # The parent alone is not a trajectory for an unmeasured arm.
            if len(rows) < 2:
                continue
            ax.plot([int(r['chats_seen']) for r in rows],[float(r[col]) for r in rows],color=st['color'],ls=st['ls'],marker='o',ms=3,label=SHORT[tid].replace('Constant · ',''))
        ax.axhline(float(ev['parent-310000'][0][col]),color=MUTED,ls=':',lw=.8,label='Pretrained parent')
        ax.set_title(title,fontsize=10,pad=10)
        ax.set(xlim=(-300,15500),ylim=limits,xticks=[0,5000,10000,15000])
        kchats(ax)
        clean(ax)
        if col != 'retention_bpb':
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.07),ncol=4,frameon=False)
    fig.text(.09,.025,"B = batch; S = seed. Open80 items are not distributed. B64 / S18 has no Open80 or raw-fact scores.",fontsize=9,color=MUTED)
    save(fig,'sft_benchmarks')


def fig_sft_endpoints() -> None:
    traj,_,ev=sft_tables()
    ids=['flat-b32-s17','flat-b64-s17','latedecay-b32-s17','latedecay-b64-s17','wd-b32-s17','wd-b32-s18','wsd-b32-s17','wsd-b32-s18']
    cols=[('val_nll','NLL ↓'),('blimp_of_64','BLiMP\n/64'),('arc_raw_of_30','ARC\n/30'),('instr_constraints_of_36','Instructions\n/36'),('stopping_native_of_16','Stopping\n/16'),('open80_native_end_of_80','Open80\n/80'),('retention_bpb','Bits/byte\n↓')]
    fig,ax=plt.subplots(figsize=(8.8,5.8))
    ax.set_axis_off()
    heading(fig,'SFT at 15,000 conversations','10,493,117 assistant targets · FP16 in-loop evaluation')
    # A numeric matrix retains precision without seven narrow, illegible axes.
    x=[.40,.49,.57,.67,.77,.86,.95]
    for xx,(_,title) in zip(x,cols):fig.text(xx,.76,title,ha='center',va='center',fontsize=9,weight='bold')
    rows=[('Pretrained parent',ev['parent-310000'][0])]+[(SHORT[tid],next(r for r in traj_evals(traj,ev,tid) if r['chats_seen']=='15000')) for tid in ids]
    for i,(label,row) in enumerate(rows):
        y=.685-i*.061
        fig.add_artist(plt.Rectangle((.035,y-.027),.95,.055,transform=fig.transFigure,fc='#f0f4f7' if i%2==0 else 'white',ec='none',zorder=0))
        label=label.replace('Warm-up + stable + decay','Warm-up / stable / decay')
        fig.text(.05,y,label,va='center',fontsize=9)
        for xx,(col,_) in zip(x,cols):
            v=float(row[col])
            txt=f'{v:.4f}' if col=='val_nll' else f'{v:.3f}' if col=='retention_bpb' else f'{v:g}'
            fig.text(xx,y,txt,ha='center',va='center',fontsize=10)
    fig.text(.05,.09,'B = batch size; S = seed. Warm-up schedules use B32. Open80 measures stopping only.',fontsize=9,color=MUTED)
    fig.text(.05,.045,'NLL and bits/byte: lower is better. Other columns: counts. One run per arm; no aggregate score.',fontsize=9,color=MUTED)
    save(fig,'sft_endpoints_15k')


def fig_sft_long() -> None:
    rows = [r for r in read_csv("sft_long_validation.csv") if r["eval_condition"] == "native"]
    fig, ax = plt.subplots(figsize=(6.2, 3.45))
    fig.subplots_adjust(left=.13, right=.96, top=.95, bottom=.18)
    arms = (
        ("system_prompt", VERMIL, "-", "Added System prompts"),
        ("no_system", BLUE, "--", "No added System prompts"),
    )
    for arm, color, style, label in arms:
        points = sorted(
            (r for r in rows if r["arm"] == arm and int(r["chats_seen"]) >= 15000),
            key=lambda row: int(row["chats_seen"]),
        )
        ax.plot([int(r["chats_seen"]) for r in points], [float(r["val_nll"]) for r in points],
                color=color, ls=style, label=label)
        endpoint = points[-1]
        ax.plot(int(endpoint["chats_seen"]), float(endpoint["val_nll"]), "o", color=color, ms=4)
    common = [r for r in rows if r["arm"] == "no_system" and int(r["chats_seen"]) <= 15000]
    ax.plot([int(r["chats_seen"]) for r in common], [float(r["val_nll"]) for r in common],
            color=INK, marker="o", ms=3, label="Shared run")
    ax.axvline(15000, color=MUTED, ls=":", lw=.8)
    ax.set(xlim=(-1000, 57500), ylim=(1.60, 2.15), ylabel="Assistant NLL (nats / target)")
    ax.set_xticks([0, 15000, 30000, 45000, 55490])
    kchats(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    clean(ax)
    save(fig, "sft_long_nll", release=True)


def main() -> None:
    fig_sft_long()
    fig_pretrain_loss()
    fig_pretrain_lr()
    fig_pretrain_themes()
    fig_sft_validation()
    fig_sft_train()
    fig_sft_lr()
    fig_sft_benchmarks()
    fig_sft_endpoints()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
