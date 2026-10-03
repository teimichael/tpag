"""Draw the two manuscript figures from released evidence and derived split rows."""

from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TIGHT = ROOT / "results_tighten"
FIG = ROOT / "outputs" / "figures"
WIDTH = 5.4
plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
SAME, CROSS = ("#c0392b", "#2471a3")
PERM, STRICT = ("#d35400", "#1e8449")


def load_json(p: Path):
    return json.loads(p.read_text())


def load_csv(p: Path) -> list[dict]:
    with p.open() as f:
        return list(csv.DictReader(f))


def independence_scatter() -> None:
    from matplotlib.lines import Line2D

    suites = [
        (
            "(a) main suite: 108 points, $n\\in\\{80,240\\}$",
            RES / "rq_e3_llm_correlation/llm_points.json",
        ),
        ("(b) re-measurement: 92 points, $n=900$", TIGHT / "rq_e3_llm_correlation/llm_points.json"),
    ]
    zero, sep = (3e-06, 1e-05)

    def z(x: float) -> float:
        return x if x > 0 else zero

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45), sharey=True)
    for ax, (title, path) in zip(axes, suites):
        pts = load_json(path)
        for div, color in (("same_model", SAME), ("cross_family", CROSS)):
            for hard, marker in (("permissive", "o"), ("strict", "^")):
                sel = [p for p in pts if p["diversity"] == div and p["hardening"] == hard]
                robust = [p for p in sel if p["independence_violated_ci"]]
                other = [p for p in sel if not p["independence_violated_ci"]]
                ax.scatter(
                    [z(p["multiplicative"]) for p in other],
                    [z(p["observed_comiss"]) for p in other],
                    s=12,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=0.7,
                )
                ax.scatter(
                    [z(p["multiplicative"]) for p in robust],
                    [z(p["observed_comiss"]) for p in robust],
                    s=12,
                    marker=marker,
                    color=color,
                    linewidths=0,
                )
        ax.plot([sep, 1], [sep, 1], color="0.35", lw=0.8, ls="--")
        ax.axvline(sep, color="0.6", lw=0.6, ls=":")
        ax.axhline(sep, color="0.6", lw=0.6, ls=":")
        ax.text(2e-05, 0.3, "co-miss above\nthe product", fontsize=6.5, color="0.3", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(zero / 2, 1.5)
        ax.set_ylim(zero / 2, 1.5)
        ticks = [zero, 0.0001, 0.001, 0.01, 0.1, 1]
        labels = ["0", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "1"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.minorticks_off()
        ax.set_xlabel("independence product $\\prod\\hat\\varepsilon_i$")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("observed co-miss rate")
    handles = [
        Line2D([], [], ls="", marker="s", color=SAME, ms=4, label="same-model"),
        Line2D([], [], ls="", marker="s", color=CROSS, ms=4, label="cross-family"),
        Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", ms=4, label="permissive"),
        Line2D([], [], ls="", marker="^", mfc="none", mec="0.3", ms=4, label="hardened"),
        Line2D([], [], ls="", marker="o", color="0.3", ms=4, label="filled: CI-robust violation"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        columnspacing=1.2,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(
        FIG / "independence_scatter.pdf",
        metadata={"Creator": "Replication package", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def validity() -> None:
    import random

    fig, (h, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.05))
    splits = [
        s
        for s in load_json(ROOT / "outputs/derived/llm_heldout_splits.json")
        if s["suite"] == "tighten"
    ]
    for hard, color in (("permissive", PERM), ("strict", STRICT)):
        sel = [s for s in splits if s["hardening"] == hard]
        h.scatter(
            [s["certificate"] for s in sel],
            [s["observed_test"] for s in sel],
            s=7,
            color=color,
            alpha=0.6,
            linewidths=0,
        )
    h.plot([0, 1], [0, 1], color="0.35", lw=0.8, ls="--")
    h.set_xlim(-0.03, 1.03)
    h.set_ylim(-0.03, 1.03)
    h.set_xlabel("certificate (600 requests)")
    h.set_ylabel("held-out co-miss (300 requests)")
    h.set_title("(a) held-out draws", loc="left")
    held = load_json(ROOT / "outputs/derived/llm_held_out_regime.json")["pairs"]
    rng = random.Random(0)
    for hard, color, jit in (("permissive", PERM, 0.012), ("strict", STRICT, 0.0)):
        sel = [p for p in held if f"|{hard}|" in p["group"]]
        xs = [p["frozen_bound"] + rng.uniform(-jit, jit) for p in sel]
        ys = [p["observed_test"] + rng.uniform(-jit, jit) for p in sel]
        b.scatter(xs, ys, s=9, color=color, alpha=0.55 if jit else 0.85, linewidths=0)
    b.plot([0, 1], [0, 1], color="0.35", lw=0.8, ls="--")
    b.set_xlim(-0.03, 1.03)
    b.set_ylim(-0.03, 1.03)
    b.set_xlabel("certificate at fraction 0")
    b.set_ylabel("co-miss at fraction 0.5 or 1")
    b.set_title("(b) injection shift", loc="left")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], ls="", marker="o", color=PERM, ms=4, label="permissive prompt"),
        Line2D([], [], ls="", marker="o", color=STRICT, ms=4, label="hardened prompt"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(
        FIG / "validity.pdf",
        metadata={"Creator": "Replication package", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.force and any(
        (FIG / name).exists() for name in ("independence_scatter.pdf", "validity.pdf")
    ):
        parser.error("figures exist; pass --force to replace them")
    FIG.mkdir(parents=True, exist_ok=True)
    independence_scatter()
    validity()
    print("wrote", sorted((p.name for p in FIG.glob("*.pdf"))))


if __name__ == "__main__":
    main()
