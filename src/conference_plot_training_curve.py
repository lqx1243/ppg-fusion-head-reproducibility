"""Arrange the same archived validation means in a single-column ACM figure."""
from pathlib import Path
import csv
import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "generated/training_validation_curve_summary.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/training_validation_curves.pdf")
    args = parser.parse_args()
    with args.input.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 240
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": "Times New Roman" if "Times New Roman" in fonts else "DejaVu Serif",
        "font.size": 8.2, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.6), sharex=True, sharey=True)
    for ax, head, title in zip(axes, ("linear", "mlp"), ("(a) Linear regression head", "(b) MLP regression head")):
        for base, color in zip(range(3), ("#305da7", "#dc8924", "#26877a")):
            values = sorted((r for r in rows if r["mode"] == f"r{base}_{head}"), key=lambda r: int(r["epoch"]))
            assert len(values) == 40 and all(int(r["n"]) == 75 for r in values)
            ax.plot([int(r["epoch"]) for r in values], [float(r["inner_mae_mean_bpm"]) for r in values],
                    color=color, linewidth=1.2, label=f"R{base}-{'L' if head == 'linear' else 'M'}")
        ax.axvline(12, color="0.5", linestyle="--", linewidth=.7)
        ax.grid(axis="y", color="0.88", linewidth=.5)
        ax.set_title(title, fontsize=9)
        ax.set(xlim=(1, 40), ylim=(7, 12.4), yticks=(7, 8, 9, 10, 11, 12))
        ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    axes[-1].set_xticks((1, 12, 20, 30, 40))
    axes[-1].set_xlabel("Training epoch")
    fig.text(.02, .53, "Mean inner-validation MAE (bpm)", rotation=90, va="center", fontsize=8.5)
    fig.subplots_adjust(left=.15, right=.97, bottom=.115, top=.93, hspace=.32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"CreationDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()
