"""Redraw the unchanged V5 signal example with a time axis on every panel."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'

import numpy as np

def signals_for_case():
    with np.load(ROOT / 'inputs/signal_example.npz') as z:
        return z['ppg'],z['acc_x'],z['saved'],z['ecg'],float(z['target'])

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figures"


def main():
    ppg, acc_x, saved, ecg, target = signals_for_case()
    font_manager.findfont(FONT)
    plt.rcParams.update({"font.family": FONT, "pdf.fonttype": 42,
                         "font.size": 8.5, "axes.titlesize": 9,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    fig = plt.figure(figsize=(6.65, 3.65), facecolor="white")
    grid = fig.add_gridspec(3, 2, left=.085, right=.98, top=.93,
                           bottom=.12, hspace=1.1, wspace=.3)
    panels = [
        (grid[0, 0], "(a) Native wrist PPG, 64 Hz", ppg, 64, "#185676", "PPG (a.u.)", (-83, 66), [-60, 0, 60]),
        (grid[0, 1], "(b) Filtered and centered PPG, 64 Hz", saved[0], 64, "#185676", "PPG (a.u.)", (-83, 66), [-60, 0, 60]),
        (grid[1, 0], "(c) Native wrist ACC x, 32 Hz", acc_x, 32, "#805524", "ACC x (g)", (-.76, .40), [-.6, 0, .3]),
        (grid[1, 1], "(d) Resampled and centered ACC x, 64 Hz", saved[1], 64, "#805524", "ACC x (g)", (-.76, .40), [-.6, 0, .3]),
        (grid[2, :], "(e) Native chest ECG, 700 Hz", ecg, 700, "#344550", "ECG (a.u.)", (-1.22, 1.20), [-1, 0, 1]),
    ]
    for position, title, values, rate, color, label, limits, ticks in panels:
        ax = fig.add_subplot(position)
        ax.plot(np.arange(len(values)) / rate, values, color=color, linewidth=.75)
        ax.set(xlim=(0, 8), ylim=limits, yticks=ticks, xticks=[0, 2, 4, 6, 8],
               xlabel="Time (s)", ylabel=label)
        ax.set_title(title, loc="left", weight="bold", pad=4)
        ax.tick_params(length=2, width=.5, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#758087")
            ax.spines[side].set_linewidth(.5)
        if rate == 700:
            ax.text(.99, .91, f"Supplied window HR: {target:.2f} bpm",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1})
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "signals.pdf", metadata={"Title": "S8 walking window 3114 with explicit time axes", "CreationDate": None})
    fig.savefig(OUT / "signals.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
