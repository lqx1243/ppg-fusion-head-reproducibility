"""Draw the two ACC fusion routes using the same saved eight-second input."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'

from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs/signal_example.npz"
OUT = ROOT / "figures/fusion_routes"
INK, GREY, FAINT = "#172936", "#55616a", "#ccd2d4"
PPG, ACC = "#185676", "#805524"
POOL_X = 87.1
HEAD_FORK_X = 129.5


def example_window():
    with np.load(INPUT) as z: return z['saved']


def label(ax, x, y, words, size=8, color=INK, **styles):
    ax.text(x, y, words, fontsize=size, color=color, va="center", **styles)


def arrow(ax, origin, end, color=GREY):
    ax.add_patch(FancyArrowPatch(origin, end, arrowstyle="-|>", mutation_scale=6.5,
                                 shrinkA=0, shrinkB=0, linewidth=.8, color=color))


def inputs(ax, window, heights):
    # The same preprocessed window is scaled per channel for display in both rows.
    names = ("PPG", "ACC x", "ACC y", "ACC z")
    for signal, name, y, color in zip(window, names, heights, (PPG, ACC, ACC, ACC)):
        label(ax, 28, y, name, size=8, color=color, ha="right")
        centered = signal - (signal.min() + signal.max()) / 2
        ax.plot(np.linspace(29.1, 47.2, signal.size),
                y + 1.26 * centered / np.max(np.abs(centered)), color=color, lw=.62)
        arrow(ax, (48.3, y), (53.0, y), color)


def mix(color, white_fraction):
    return np.asarray(to_rgb(color)) * (1 - white_fraction) + white_fraction


def prism(ax, x, y, width, height, depth, color):
    """A front face and two receding faces. Heights illustrate temporal reduction."""
    lo, hi = y - height / 2, y + height / 2
    dx, dy = depth, depth * .55
    edge = mix(color, .08)
    faces = [
        ([(x, hi), (x + dx, hi + dy), (x + width + dx, hi + dy), (x + width, hi)], .74),
        ([(x + width, lo), (x + width + dx, lo + dy),
          (x + width + dx, hi + dy), (x + width, hi)], .15),
        ([(x, lo), (x + width, lo), (x + width, hi), (x, hi)], .46),
    ]
    for points, light in faces:
        ax.add_patch(Polygon(points, closed=True, fc=mix(color, light), ec=edge,
                             linewidth=.45, zorder=3, joinstyle="round"))


def encoder(ax, y, first_height, color):
    # Each stack stands for one convolution/ReLU stage; all routes have four.
    x = 53.0
    ax.plot([x, POOL_X - 2], [y, y], color=mix(color, .45), lw=.8, zorder=1)
    for scale, width, depth in zip((1, .73, .52, .36),
                                   (1.65, 2.1, 2.7, 2.7), (1.7, 2.1, 2.65, 2.65)):
        prism(ax, x, y, width, first_height * scale, depth, color)
        # Measure the 3.4 mm gap from the receding face, not the block's origin.
        x += width + depth + 3.4
    arrow(ax, (POOL_X - 4.9, y), (POOL_X - .4, y), color)


def vector(ax, x, y, width, value, color):
    ax.add_patch(Rectangle((x, y - 1.9), width, 3.8, ec=color,
                           fc=mix(color, .79), lw=.65, zorder=4))
    label(ax, x + width / 2, y, value, size=8, ha="center", zorder=5)


def readout(ax, y):
    # Each alternative ends at its own scalar output, so neither route implies an ensemble.
    upper, lower = y + 3.35, y - 3.35
    ax.plot([132, HEAD_FORK_X, HEAD_FORK_X, 132], [upper, upper, lower, lower],
            color=GREY, lw=.65)
    label(ax, 133, upper, "Linear", size=8.3, weight="bold")
    arrow(ax, (142.5, upper), (165, upper))
    label(ax, 133, lower, "MLP", size=8.3, weight="bold")
    arrow(ax, (140.3, lower), (145.2, lower))
    ax.add_patch(Rectangle((145.6, lower - 2.1), 13.5, 4.2, facecolor="#eaf0f3",
                           edgecolor=GREY, linewidth=.65))
    label(ax, 152.35, lower, "32 · ReLU", size=8, ha="center")
    arrow(ax, (159.6, lower), (165, lower))
    for level in (upper, lower):
        label(ax, 166.2, level, "HR", size=8.5, weight="bold")


def main():
    global INPUT, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUT.parent)
    args = parser.parse_args()
    INPUT, OUT = args.input, args.output / "fusion_routes"
    font_manager.findfont(FONT)
    plt.rcParams.update({"font.family": FONT, "svg.fonttype": "none",
                         "svg.hashsalt": "ppg-v5-fusion-routes", "pdf.fonttype": 42})
    window = example_window()
    width, height = 174, 55  # Logical drawing coordinates.
    # Compact vertical geometry, retaining 8--9 point labels at the printed width.
    fig = plt.figure(figsize=(7.0, 49 / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_axis_off()
    for x, title in [(21, "One 8-s window"), (53, "4 convolutional stages"),
                     (POOL_X, "After average pooling"), (HEAD_FORK_X, "Choose one head")]:
        label(ax, x, 52.2, title, size=8.7, weight="bold")
    ax.plot([1, 172], [49.5, 49.5], color=FAINT, lw=.6)
    label(ax, 1, 41, "(a)  Early", weight="bold", size=9)
    label(ax, 1, 37.2, "fusion", weight="bold", size=9)
    inputs(ax, window, (45.2, 41.2, 37.2, 33.2))
    joint = "#415f78"
    encoder(ax, 39.2, 14, joint)
    vector(ax, POOL_X, 39.2, 13, "64", joint)
    arrow(ax, (POOL_X + 13.3, 39.2), (HEAD_FORK_X, 39.2))
    readout(ax, 39.2)

    ax.plot([1, 172], [30.7, 30.7], color=FAINT, lw=.45)
    label(ax, 1, 20, "(b)  Feature", weight="bold", size=9)
    label(ax, 1, 16.2, "fusion", weight="bold", size=9)
    ppg_y, acc_y, joined_y = 24.5, 12.7, 18.6
    inputs(ax, window, (ppg_y, acc_y + 3.8, acc_y, acc_y - 3.8))
    for y, color in ((ppg_y, PPG), (acc_y, ACC)):
        encoder(ax, y, 9.2, color)
        vector(ax, POOL_X, y, 10.5, "44", color)

    # Each branch retains its own 44 coordinates in the concatenated vector.
    ax.plot([POOL_X + 10.5, 107.3], [ppg_y, ppg_y], color=PPG, lw=.8)
    arrow(ax, (107.3, ppg_y), (111.9, joined_y + 1.9), PPG)
    ax.plot([POOL_X + 10.5, 113.8], [acc_y, acc_y], color=ACC, lw=.8)
    arrow(ax, (113.8, acc_y), (118.4, joined_y - 1.9), ACC)
    vector(ax, 108.7, joined_y, 6.5, "", PPG)
    vector(ax, 115.2, joined_y, 6.5, "", ACC)
    label(ax, 115.2, 25.8, "Concatenate", ha="center", size=8)
    label(ax, 115.2, 9.1, "88 features", ha="center", size=8)
    arrow(ax, (122, joined_y), (HEAD_FORK_X, joined_y))
    readout(ax, joined_y)

    ax.plot([1, 172], [6.4, 6.4], color=FAINT, lw=.65)
    label(ax, 1, 2.8, "PPG-only control", weight="bold", size=8)
    label(ax, 23, 2.8, "One PPG encoder → 64 features → either head.", size=8.4)
    label(ax, 88, 2.8, "Six models", weight="bold", size=8)
    label(ax, 103.5, 2.8, "Three input designs × two heads, trained end to end.", size=8.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = OUT.with_suffix(".svg")
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT.with_suffix(".pdf"), metadata={"CreationDate": None})
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
