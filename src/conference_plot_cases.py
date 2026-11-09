"""Replay the disclosed post-hoc Figure 5 selection and redraw selected behaviors."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager
FONT = 'Times New Roman' if any(f.name == 'Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'

ROOT=Path(__file__).resolve().parent;INPUTS=ROOT/'inputs';OUT=ROOT/'figures'
MODES=('r0_mlp','r1_mlp','r2_mlp');LABELS=('PPG only','Early fusion','Feature fusion')
def draw(plots):
    font_manager.findfont(FONT)
    plt.rcParams.update({"font.family": FONT, "font.size": 8.4, "pdf.fonttype": 42,
                         "axes.titlesize": 9.1, "axes.labelsize": 8.6, "axes.linewidth": .55,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#55616a", "text.color": "#172936",
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    fig, axes = plt.subplots(2, 1, figsize=(7, 2.9), sharex=True)
    fig.subplots_adjust(left=.068, right=.977, top=.865, bottom=.16, hspace=.44)
    styles = (("#185676", (0, (5, 2.3))), ("#326b60", (0, (1.2, 1.5))), ("#914912", (0, (4, 1.5, .8, 1.5))))
    for ax, letter, (case, x, truth, pred) in zip(axes, ("a", "b"), plots):
        ax.plot(x, truth, color="#181818", linewidth=1.65, label="ECG-derived HR", zorder=5)
        for mode, label, (color, linestyle) in zip(MODES, LABELS, styles):
            ax.plot(x, pred[mode], color=color, ls=linestyle, linewidth=1.15, label=label, zorder=3)
        ax.text(0, 1.07, f"({letter})  S{case['subject']}: walking", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=9)
        maes = [case[m + "_mae_bpm"] for m in MODES]
        ax.text(1, 1.07, "MAE (PPG / early / feature, bpm): " + " / ".join(f"{v:.2f}" for v in maes),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.1)
        ax.grid(axis="y", color="#d8dfe1", linewidth=.45, zorder=0)
        values = np.concatenate([truth, *pred.values()])
        ax.set(xlim=(0, 120), ylim=(10 * np.floor((values.min() - 3) / 10), 10 * np.ceil((values.max() + 3) / 10)),
               xticks=np.arange(0, 121, 20), ylabel="Heart rate (bpm)")
        ax.yaxis.set_major_locator(MultipleLocator(20))
    axes[-1].set_xlabel("Time within the 120-s segment (s)", labelpad=4)
    fig.legend(*axes[0].get_legend_handles_labels(), ncols=4, loc="upper center", bbox_to_anchor=(.53, .998),
               frameon=False, handlelength=3, columnspacing=1.8, borderpad=0)
    for suffix in ("pdf", "png"):
        options = {"metadata": {"Title": "Selected 120-second walking excerpts", "CreationDate": None}} if suffix == "pdf" else {"dpi": 220}
        fig.savefig(OUT / f"cases.{suffix}", **options)
    plt.close(fig)

def main():
    global INPUTS,OUT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=INPUTS);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    INPUTS,OUT=a.inputs,a.output
    provenance=json.loads((INPUTS/'case_selection.json').read_text())
    with (INPUTS/'case_walking_predictions.csv').open() as f: rows=list(csv.DictReader(f))
    plots=[]
    for case in provenance['cases']:
        person=case['subject']; pool=sorted([r for r in rows if int(r['subject'])==person],key=lambda r:int(r['window']))
        candidates=[]
        for i in range(len(pool)-56):
            seg=pool[i:i+57]; windows=np.array([int(r['window']) for r in seg])
            if not np.array_equal(windows, np.arange(windows[0],windows[0]+57)): continue
            truth=np.array([float(r['reference_bpm']) for r in seg]); pred={m:np.clip([float(r[m+'_raw_bpm']) for r in seg],30,220) for m in MODES}
            errors={m:float(np.abs(v-truth).mean()) for m,v in pred.items()}
            candidates.append((errors[case['selection_model']],int(windows[0]),truth,pred,errors))
        assert len(candidates)==case['eligible_candidates']
        best=min(candidates,key=lambda x:(x[0],x[1]));assert best[1]==case['start_window']
        for m in MODES: np.testing.assert_allclose(best[4][m],case[m+'_mae_bpm'],rtol=0,atol=1e-12)
        plots.append((case,np.arange(4,117,2),best[2],best[3]))
    OUT.mkdir(parents=True,exist_ok=True);draw(plots)
if __name__=='__main__': main()
