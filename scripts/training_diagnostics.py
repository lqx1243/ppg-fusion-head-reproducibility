#!/usr/bin/env python3
"""Replay exported selection histories and regenerate training diagnostics.

No Torch import, model forward or training. CSV replay verifies the full grid and
strict checkpoint rule; it does not reverify original private record hashes.
Figures require Matplotlib. Defaults use the archived independent inputs.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/training'
FIGURES = ROOT / 'outputs/figures'
PROTOCOL_SHA = 'bfc78984f4dd4cd7e25158439b267161475be0e71972cff96109e3594d20aee5'
MODES = ('r0_linear', 'r0_mlp', 'r1_linear', 'r1_mlp', 'r2_linear', 'r2_mlp')
LABELS = dict(zip(MODES, ('R0-L', 'R0-M', 'R1-L', 'R1-M', 'R2-L', 'R2-M')))
SEEDS = tuple(range(3001, 3006))
ENDPOINTS = (12, 40)
IMPROVEMENT_MIN = 0.0001
NEAR_40_START = 36  # Descriptive definition, inclusive final five epochs.
HISTORY_FIELDS = ('mode', 'seed', 'person', 'epoch', 'train_loss', 'inner_mae_bpm', 'train_seconds', 'inner_seconds')
SELECTED_FIELDS = ('endpoint', 'mode', 'seed', 'person', 'selected_epoch',
                   'selected_inner_float32_mae_bpm', 'prefix_last_inner_float32_mae_bpm', 'prefix_last_train_loss')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, fraction):
    """Linear interpolation on sorted values, matching NumPy's default."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)




def validate_rows(histories, selections):
    expected = {(mode, seed, person) for mode in MODES for seed in SEEDS for person in range(1, 16)}
    groups, chosen = {}, {}
    for row in histories:
        for key in ('seed', 'person', 'epoch'):
            row[key] = int(row[key])
        for key in HISTORY_FIELDS[4:]:
            row[key] = float(row[key])
            require(math.isfinite(row[key]) and row[key] >= 0, f'invalid history value: {key}')
        key = (row['mode'], row['seed'], row['person'])
        groups.setdefault(key, []).append(row)
    for row in selections:
        for field in ('endpoint', 'seed', 'person', 'selected_epoch'):
            row[field] = int(row[field])
        for field in SELECTED_FIELDS[5:]:
            row[field] = float(row[field])
            require(math.isfinite(row[field]) and row[field] >= 0, f'invalid selected value: {field}')
        key = (row['mode'], row['seed'], row['person'], row['endpoint'])
        require(key not in chosen, 'duplicate selected endpoint')
        chosen[key] = row
    require(set(groups) == expected and len(histories) == 18000, 'incomplete history grid')
    require(set(chosen) == {(*key, endpoint) for key in expected for endpoint in ENDPOINTS}, 'incomplete selected grid')
    for key, rows in groups.items():
        rows.sort(key=lambda row: row['epoch'])
        require([row['epoch'] for row in rows] == list(range(1, 41)), 'incomplete ordered 40-epoch history')
        best, best_epoch = math.inf, 0
        for row in rows:
            if row['inner_mae_bpm'] < best - IMPROVEMENT_MIN:
                best, best_epoch = row['inner_mae_bpm'], row['epoch']
            if row['epoch'] in ENDPOINTS:
                selected = chosen[*key, row['epoch']]
                require(selected['selected_epoch'] == best_epoch and selected['selected_inner_float32_mae_bpm'] == best,
                        f'strict-selection replay differs {key} endpoint {row["epoch"]}')
                require(selected['prefix_last_inner_float32_mae_bpm'] == row['inner_mae_bpm']
                        and selected['prefix_last_train_loss'] == row['train_loss'], 'endpoint last-epoch values differ')
    return groups, chosen


def summarize(groups, chosen):
    summaries, curves, improvements = [], [], []
    for mode in MODES:
        keys = [(mode, seed, person) for seed in SEEDS for person in range(1, 16)]
        for endpoint in ENDPOINTS:
            epochs = [chosen[*key, endpoint]['selected_epoch'] for key in keys]
            summaries.append(dict(mode=mode, label=LABELS[mode], endpoint=endpoint, n=75,
                median=median(epochs), q1=quantile(epochs, .25), q3=quantile(epochs, .75),
                minimum=min(epochs), maximum=max(epochs), at_endpoint=sum(x == endpoint for x in epochs),
                near40_36_to_40=sum(x >= NEAR_40_START for x in epochs),
                near40_percent=100 * sum(x >= NEAR_40_START for x in epochs) / 75,
                after_epoch12=sum(x > 12 for x in epochs)))
        for epoch in range(1, 41):
            vals = [groups[key][epoch - 1]['inner_mae_bpm'] for key in keys]
            losses = [groups[key][epoch - 1]['train_loss'] for key in keys]
            curves.append(dict(mode=mode, label=LABELS[mode], epoch=epoch, n=75,
                inner_mae_mean_bpm=mean(vals), inner_mae_median_bpm=median(vals),
                inner_mae_q1_bpm=quantile(vals, .25), inner_mae_q3_bpm=quantile(vals, .75),
                train_loss_mean=mean(losses)))
        changes = [chosen[*key, 40]['selected_inner_float32_mae_bpm'] - chosen[*key, 12]['selected_inner_float32_mae_bpm'] for key in keys]
        raw_changes = [groups[key][39]['inner_mae_bpm'] - groups[key][34]['inner_mae_bpm'] for key in keys]
        best35 = []
        for key in keys:
            best = math.inf
            for row in groups[key][:35]:
                if row['inner_mae_bpm'] < best - IMPROVEMENT_MIN:
                    best = row['inner_mae_bpm']
            best35.append(best)
        late_changes = [chosen[*key, 40]['selected_inner_float32_mae_bpm'] - best
                        for key, best in zip(keys, best35)]
        improvements.append(dict(mode=mode, n=75,
            selected_inner_mae_40_minus_12_mean_bpm=mean(changes),
            selected_inner_mae_40_minus_12_median_bpm=median(changes),
            selected_inner_mae_improved_after12=sum(value < -IMPROVEMENT_MIN for value in changes),
            selected_inner_mae_40_minus35_mean_bpm=mean(late_changes),
            selected_inner_mae_40_minus35_median_bpm=median(late_changes),
            selected_inner_mae_improved_after35=sum(value < -IMPROVEMENT_MIN for value in late_changes),
            raw_inner_mae_epoch40_minus35_mean_bpm=mean(raw_changes)))
    return summaries, curves, improvements


def table_tex(summaries):
    indexed = {(row['mode'], row['endpoint']): row for row in summaries}
    result = [r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
        r' & 1--12 & \multicolumn{4}{c}{Selection within epochs 1--40} \\',
        r'\cmidrule(l){3-6}',
        r'Configuration & Median & Median [IQR] & Range & 36--40 & At 40 \\', r'\midrule']
    for mode in MODES:
        a, b = indexed[mode, 12], indexed[mode, 40]
        result.append(f'{LABELS[mode]} & {a["median"]:g} & {b["median"]:g} [{b["q1"]:g}, {b["q3"]:g}] & '
                      f'{b["minimum"]}--{b["maximum"]} & {b["near40_36_to_40"]}/75 & {b["at_endpoint"]}/75' + r' \\')
    return '\n'.join(result + [r'\bottomrule', r'\end{tabular}', ''])


def plot_curves(curves, figure_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7), sharex=True, sharey=True, layout='constrained')
    colors = ('#315a9d', '#d1852d', '#248276')
    for ax, head, title in zip(axes, ('linear', 'mlp'), ('Linear regression head', 'MLP regression head')):
        for base, color in zip(range(3), colors):
            mode = f'r{base}_{head}'
            rows = [row for row in curves if row['mode'] == mode]
            ax.plot([r['epoch'] for r in rows], [r['inner_mae_mean_bpm'] for r in rows],
                    color=color, linewidth=1.6, label=LABELS[mode])
        ax.axvline(12, color='.5', linewidth=.8, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Training epoch')
        ax.set_xticks([1, 12, 20, 30, 40])
        ax.set_xlim(1, 40)
        ax.grid(axis='y', color='.88', linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    axes[0].set_ylabel('Mean inner-validation MAE (bpm)')
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(figure_dir / f'training_validation_curves.{suffix}', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-csv', type=Path, default=ROOT/'inputs/training_history.csv')
    parser.add_argument('--selected-csv', type=Path, default=ROOT/'inputs/training_selected_epochs.csv')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--figure-dir', type=Path, default=FIGURES)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    require(bool(args.history_csv) == bool(args.selected_csv), 'portable mode requires both CSVs')
    if args.history_csv:
        histories, selections = read_csv(args.history_csv), read_csv(args.selected_csv)
        provenance = {'mode': 'portable_csv', 'original_record_hashes_verified': False,
            'history_csv_sha256': sha(args.history_csv), 'selected_csv_sha256': sha(args.selected_csv)}
    else:
        raise ValueError('Both history and selected CSVs are required')
    groups, chosen = validate_rows(histories, selections)
    summaries, curves, improvements = summarize(groups, chosen)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'training_history.csv', histories, HISTORY_FIELDS)
    write_csv(output / 'training_selected_epochs.csv', selections, SELECTED_FIELDS)
    write_csv(output / 'training_epoch_summary.csv', summaries)
    write_csv(output / 'training_validation_curve_summary.csv', curves)
    write_csv(output / 'training_validation_changes.csv', improvements)
    (output / 'training_epochs_table.tex').write_text(table_tex(summaries))
    audit = {'schema_version': 1, 'trajectories': len(groups), 'epochs': len(histories), 'endpoints': len(chosen),
        'all_strict_selections_replayed': True, 'selection_improvement_min_bpm': IMPROVEMENT_MIN,
        'near40_definition': 'post hoc descriptive audit definition: selected epoch in inclusive range 36--40 (final five epochs)',
        'curve_aggregation': 'equal arithmetic mean of 75 per-fit pooled-window inner-validation MAEs per configuration and epoch; no smoothing; no outer per-epoch evaluation',
        'quantiles': 'linear interpolation; IQR denotes 25th and 75th percentiles, descriptive across 75 dependent fits',
        'limitations': ['Five seeds and overlapping participant-held-out folds are dependent, not 75 independent participants.',
                       'Selected epochs are validation minima under a bounded selection rule; they do not establish convergence.',
                       'Different selected epochs and validation learning curves cannot separate learning speed, capacity, and generalization.',
                       'No observations beyond epoch 40; no new training or model forward.'],
        'provenance': provenance, 'epoch_summaries': summaries, 'validation_changes': improvements}
    (output / 'training_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if not args.no_plot:
        plot_curves(curves, args.figure_dir)
    print(json.dumps({'trajectories': len(groups), 'epochs': len(histories), 'selected_endpoints': len(chosen),
                      'epoch40_summaries': [row for row in summaries if row['endpoint'] == 40]}, indent=2))


if __name__ == '__main__':
    main()
