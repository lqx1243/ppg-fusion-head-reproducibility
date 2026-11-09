"""Draw the V5 results from saved summaries, fixed case IDs and archived predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'

from matplotlib.ticker import MultipleLocator

HEADS=(("linear","fusion_linear"),("mlp","primary_fusion_mlp"))
def signed(value): return f"{value:+.2f}".replace("-","−")


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/figures'
SOURCE = ROOT / 'outputs/summaries/core'
BLUE, GRAY, INK, LIGHT = "#185676", "#55616a", "#172936", "#d8dfe1"
MM = 1 / 25.4


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", metadata={"Title": name.capitalize(), "CreationDate": None})
    plt.close(fig)


def interaction():
    people = participant_samples()
    with (SOURCE / "contrasts_by_seed.csv").open(newline="") as stream:
        cohort = {(int(r["endpoint"]), r["contrast"], int(r["seed"])): float(r["mae_bpm"])
                  for r in csv.DictReader(stream) if r["weighting"] == "participant_equal"}
    # Both budgets use the same horizontal ruler. The mean labels have their own
    # columns outside the data axes so a label cannot cross a cap or a connector.
    width, height = 157, 52
    fig = plt.figure(figsize=(width * MM, height * MM))
    left, bottom, plot_w, plot_h = 28, 11, 44, 29
    y_low, y_high = -.52, 2.42
    rows = np.array([2., 1., -.16])

    def page_y(y):
        return (bottom + plot_h * (y - y_low) / (y_high - y_low)) / height

    fig.text(.012, .91, "Feature − early", va="center", weight="bold", size=8.6)
    for y, label in zip(rows, ("Linear head", "MLP head", "Interaction")):
        fig.text(.012, page_y(y), label, va="center",
                 weight="bold" if y == rows[-1] else "normal", size=8.8)

    for index, (budget, color, marker, fill) in enumerate(
            ((12, GRAY, "s", "white"), (40, BLUE, "o", BLUE))):
        x = left + index * 67
        ax = fig.add_axes([x / width, bottom / height, plot_w / width, plot_h / height])
        mean_col = (x + plot_w + 8) / width
        fig.text((x + plot_w / 2) / width, .91, f"Epochs 1–{budget}", ha="center",
                 va="center", weight="bold", color=color, size=9.5)
        fig.text(mean_col, .91, "Mean", ha="center", va="center", size=8.1, color=GRAY)

        head_gaps = [people[budget, name].mean(axis=0) for _, name in HEADS]
        samples = [*head_gaps, head_gaps[1] - head_gaps[0]]
        names = [HEADS[0][1], HEADS[1][1], "secondary_head_by_fusion"]
        means = np.array([values.mean() for values in samples])
        for name, values in zip(names, samples):
            stored = [cohort[budget, name, seed] for seed in range(3001, 3006)]
            np.testing.assert_allclose(values, stored, rtol=0, atol=1e-12)

        ax.axvline(0, color="#89959b", linewidth=.65, zorder=0)
        ax.axhline(.42, color=LIGHT, lw=.55, zorder=0)
        ax.plot(means[:2], rows[:2], color=color, alpha=.62, lw=.9, zorder=1)
        for y, mean, values in zip(rows, means, samples):
            ax.errorbar(mean, y, xerr=[[mean - values.min()], [values.max() - mean]],
                        fmt=marker, color=color, mfc=fill, ms=4.5, elinewidth=.9,
                        capsize=3.4, capthick=.85, zorder=3)
            fig.text(mean_col, page_y(y), signed(mean), ha="center", va="center",
                     color=color, size=8.7, weight="bold" if y == rows[-1] else "normal")
        ax.set(xlim=(-1.15, 1.05), ylim=(y_low, y_high), yticks=[])
        ax.set_xticks([-1, -.5, 0, .5, 1], ["−1", "−0.5", "0", "+0.5", "+1"])
        ax.spines["left"].set_visible(False)
    fig.text(.52, .043, "Paired MAE difference (bpm)", ha="center", va="center", size=8.7)
    save(fig, "interaction")


def participant_samples():
    """Return 15 × 5 arrays of differences paired within participant and seed."""
    with (SOURCE / "participant_contrasts.csv").open(newline="") as stream:
        groups = {}
        for row in csv.DictReader(stream):
            key = int(row["endpoint"]), row["contrast"]
            values = [float(row[f"seed_{seed}_mae_bpm"]) for seed in range(3001, 3006)]
            groups.setdefault(key, {})[int(row["person"])] = values
    return {key: np.array([people[p] for p in range(1, 16)]) for key, people in groups.items()}


def participants():
    samples = participant_samples()
    panels = (("primary_fusion_mlp", "(a)  Feature minus early fusion"),
              ("secondary_acc_r2_mlp", "(b)  Feature fusion minus PPG only"))
    ranges = [np.ptp(samples[40, name]) for name, _ in panels]
    people = np.arange(1, 16)
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(160 * MM, 69 * MM),
                             gridspec_kw={"width_ratios": ranges})
    fig.subplots_adjust(left=.054, right=.989, top=.89, bottom=.177, wspace=.12)
    for ax, (name, title), extent in zip(axes, panels, ranges):
        seeds = samples[40, name]
        values = seeds.mean(axis=1)
        limits = (seeds.min() - .05 * extent, seeds.max() + .05 * extent)
        ax.axhspan(7.5, 8.5, color="#eaf0f2", zorder=0)
        ax.hlines(people, *limits, colors=LIGHT, lw=.35, zorder=0)
        ax.axvline(0, color=GRAY, lw=.7)
        errors = np.stack((values - seeds.min(axis=1), seeds.max(axis=1) - values))
        for mask, marker, fill in ((values < 0, "o", BLUE), (values >= 0, "D", "white")):
            ax.errorbar(values[mask], people[mask], xerr=errors[:, mask], fmt=marker,
                        color=INK, ecolor=GRAY, mfc=fill, ms=3.15, mew=.68,
                        elinewidth=.8, capsize=1.8, capthick=.8, zorder=3)
        ax.set(xlim=limits, ylim=(15.58, .42))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_title(title, loc="left", weight="bold", pad=4)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=7.5)
    axes[0].set_yticks(people, [f"S{p}" for p in people])
    axes[0].get_yticklabels()[7].set_fontweight("bold")
    fig.supxlabel("Paired MAE difference (bpm)", fontsize=8.5, y=.016)
    save(fig, "participants")



def main():
    builders = {"interaction": interaction, "participants": participants}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", nargs="?", choices=["all", *builders], default="all")
    args = parser.parse_args()
    font_manager.findfont(FONT)
    plt.rcParams.update({"font.family": FONT, "font.size": 8, "pdf.fonttype": 42,
                         "axes.titlesize": 9, "axes.labelsize": 8.5, "axes.linewidth": .55,
                         "axes.edgecolor": GRAY, "axes.labelcolor": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "xtick.color": INK, "ytick.color": INK, "xtick.major.width": .55,
                         "ytick.major.width": .55, "xtick.major.size": 2, "ytick.major.size": 2})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in builders.items():
        if args.figure in ("all", name):
            builder()


if __name__ == "__main__":
    main()
