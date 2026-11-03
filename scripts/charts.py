"""The three README charts, drawn from the rows built by make_tables.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LORA_C, GUIDED_C, PROMPT_C = "#2a78d6", "#eb6834", "#8a8984"


def _axes(title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel("accuracy (%)", color=MUTED, fontsize=9)
    return fig, ax


def _point(ax, x, r, color, label, dx=6, dy=0, size=7):
    m, lo, hi = (100 * v for v in r["official_exec"])
    ax.errorbar(
        x,
        m,
        yerr=[[m - lo], [hi - m]],
        fmt="o",
        color=color,
        ms=size,
        lw=1.5,
        capsize=0,
        mec=SURFACE,
        mew=1.5,
        zorder=3,
    )
    ax.annotate(
        label,
        (x, m),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=INK,
        va="center",
    )


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def size_chart(by: dict, out: Path) -> None:
    fig, ax = _axes("Accuracy vs model size", "parameters (billions)")
    ax.set_ylabel("accuracy (%, y-axis cropped)", color=MUTED, fontsize=9)
    prompted = [(4, "qwen3-4b-fewshot"), (8, "qwen3-8b-fewshot"), (14, "qwen3-14b-awq-fewshot")]
    ax.plot(
        [x for x, _ in prompted],
        [100 * by[t]["official_exec"][0] for _, t in prompted],
        color=PROMPT_C,
        lw=1.5,
        zorder=2,
    )
    for x, t in prompted:
        _point(ax, x, by[t], PROMPT_C, "prompted" if x == 8 else "", dx=8, size=6)
    for x, t in ((4, "qwen3-4b-fewshot-guided"), (14, "qwen3-14b-awq-fewshot-guided")):
        _point(ax, x, by[t], GUIDED_C, "prompted + guided" if x == 14 else "", dx=8, dy=4, size=6)
    _point(ax, 4, by["qwen3-4b-lora-full-ep2"], LORA_C, "")
    _point(ax, 4.45, by["qwen3-4b-lora-full-ep2-seed1"], LORA_C, "4B + LoRA (runs 1 and 2)", dx=8)
    ax.set_xticks([4, 8, 14], ["4B", "8B", "14B"])
    ax.set_xlim(2.5, 16.5)
    ax.set_ylim(48, 82)
    _save(fig, out)


def curve_chart(by: dict, curve: list, out: Path) -> None:
    fig, ax = _axes("Accuracy vs training examples", "training examples (log scale)")
    ax.set_ylabel("accuracy (%, y-axis cropped)", color=MUTED, fontsize=9)
    xs = [n for n, _ in curve]
    ms = [100 * by[t]["official_exec"][0] for _, t in curve]
    los = [100 * by[t]["official_exec"][1] for _, t in curve]
    his = [100 * by[t]["official_exec"][2] for _, t in curve]
    ax.fill_between(xs, los, his, color=LORA_C, alpha=0.15, lw=0)
    ax.plot(xs, ms, "o-", color=LORA_C, lw=2, ms=7, mec=SURFACE, mew=1.5, zorder=3)
    for x, m in zip(xs, ms, strict=True):
        ax.annotate(
            f"{m:.1f}",
            (x, m),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    for tag, color, label in (
        ("qwen3-14b-awq-fewshot-guided", GUIDED_C, "Qwen3-14B prompted + guided"),
        ("qwen3-14b-awq-fewshot", PROMPT_C, "Qwen3-14B prompted"),
    ):
        y = 100 * by[tag]["official_exec"][0]
        ax.axhline(y, color=color, lw=1.2, ls="--", zorder=1)
        ax.annotate(
            f"{label}: {y:.1f}",
            (xs[-1], y),
            xytext=(0, -11 if "guided" not in tag else 4),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            color=MUTED,
        )
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_ylim(62, 82)
    _save(fig, out)


def latency_chart(by: dict, out: Path) -> None:
    fig, ax = _axes("Accuracy vs latency (batch size 1)", "p50 latency per question (s)")
    pts = [
        ("qwen3-4b-zeroshot", PROMPT_C, "4B zero-shot", 6, 0),
        ("qwen3-4b-fewshot", PROMPT_C, "4B, 4 shots", 6, -6),
        ("qwen3-8b-fewshot", PROMPT_C, "8B, 4 shots", -6, 0),
        ("qwen3-14b-awq-fewshot", PROMPT_C, "14B AWQ, 4 shots", 8, -6),
        ("qwen3-4b-fewshot-guided", GUIDED_C, "4B guided", 6, 6),
        ("qwen3-14b-awq-fewshot-guided", GUIDED_C, "14B AWQ guided", 8, 6),
        ("qwen3-4b-lora-full-ep2", LORA_C, "4B + LoRA", 8, 4),
    ]
    for tag, color, label, dx, dy in pts:
        _point(ax, by[tag]["latency_p50_s"], by[tag], color, label, dx=dx, dy=dy)
        if dx < 0:
            ax.texts[-1].set_ha("right")
    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 82)
    _save(fig, out)
