#!/usr/bin/env python3
"""Reproduce study tables from the inputs shipped alongside this script (standard library only)."""

import argparse
import csv
import json
from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, median, pstdev

MODES = ("r0_linear", "r0_mlp", "r1_linear", "r1_mlp", "r2_linear", "r2_mlp")
SEEDS = tuple(range(3001, 3006))
RANGES = (12, 40)
WEIGHTS = ("participant_equal", "window_weighted")
METRICS = (
    "mae_bpm",
    "rmse_bpm",
    "bias_bpm",
    "within5_pct",
    "within10_pct",
    "fail15_pct",
    "coverage_pct",
    "raw_out_of_range_pct",
)


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return dict(mean=mean(values), sd_ddof0=pstdev(values), min=min(values), max=max(values))


def aggregate(cells, weighting):
    """Combine disjoint cells, pooling squared error when windows get equal weight."""
    valid = [cell for cell in cells if int(cell["n"])]
    weights = [int(cell["n"]) if weighting == "window_weighted" else 1 for cell in valid]
    total = sum(weights)
    result = {"n": sum(int(cell["n"]) for cell in valid)}
    for metric in METRICS:
        values = [float(cell[metric]) for cell in valid]
        if metric == "rmse_bpm" and weighting == "window_weighted":
            result[metric] = sqrt(
                sum(weight * value * value for weight, value in zip(weights, values)) / total
            )
        else:
            result[metric] = sum(weight * value for weight, value in zip(weights, values)) / total
    result["raw_out_of_range_n"] = sum(int(cell["raw_out_of_range_n"]) for cell in valid)
    return result


def seed_summary(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, cells in sorted(groups.items()):
        assert sorted(cell["seed"] for cell in cells) == list(SEEDS)
        item = dict(zip(keys, group), n=cells[0]["n"], n_people=cells[0]["n_people"])
        for metric in METRICS:
            item.update(
                {
                    metric + "_" + key: value
                    for key, value in stats([cell[metric] for cell in cells]).items()
                }
            )
        result.append(item)
    return result


def calculate(inputs):
    # Rebuild person-level and activity-level scores from disjoint window groups.
    person_activity = load_csv(inputs / "person_activity_metrics.csv")
    people = defaultdict(list)
    activities = defaultdict(list)
    names = {}
    seen = set()
    for row in person_activity:
        budget, mode, seed = int(row["endpoint"]), row["mode"], int(row["seed"])
        person, code = int(row["subject"]), int(row["activity_code"])
        key = (budget, mode, seed, person, code)
        assert key not in seen and budget in RANGES and mode in MODES and seed in SEEDS
        assert person in range(1, 16) and code in range(9)
        assert (int(row["n"]) == 0) == (person == 6 and code in (6, 7, 8))
        seen.add(key)
        people[budget, mode, seed, person].append(row)
        activities[budget, code, mode, seed].append(row)
        names[code] = row["activity_name"]
    assert len(seen) == 8100
    person_metrics = {key: aggregate(rows, "window_weighted") for key, rows in people.items()}
    activity = []
    for (budget, code, mode, seed), cells in sorted(activities.items()):
        for weight in WEIGHTS:
            activity.append(
                dict(
                    endpoint=budget,
                    activity_code=code,
                    activity_name=names[code],
                    mode=mode,
                    seed=seed,
                    weighting=weight,
                    n_people=sum(int(cell["n"]) > 0 for cell in cells),
                    **aggregate(cells, weight),
                )
            )
    overall = []
    for budget in RANGES:
        for mode in MODES:
            for seed in SEEDS:
                subjects = [person_metrics[budget, mode, seed, person] for person in range(1, 16)]
                for weight in WEIGHTS:
                    result = aggregate(subjects, weight)
                    assert result["n"] == 64697
                    overall.append(
                        dict(
                            endpoint=budget,
                            mode=mode,
                            seed=seed,
                            weighting=weight,
                            n_people=15,
                            **result,
                        )
                    )
    s1 = seed_summary(overall, ("endpoint", "mode", "weighting"))
    s5 = seed_summary(activity, ("endpoint", "activity_code", "activity_name", "mode", "weighting"))
    person_maes = {key: row["mae_bpm"] for key, row in person_metrics.items()}

    def paired_row(name, budget, terms):
        # Each term identifies a model, selection range and sign in the contrast.
        # Pair within person and seed, then give the 15 people equal weight.
        seed_means = []
        for seed in SEEDS:
            person_differences = []
            for person in range(1, 16):
                difference = sum(
                    sign * person_maes[selection_range, mode, seed, person]
                    for selection_range, mode, sign in terms
                )
                person_differences.append(difference)
            seed_means.append(mean(person_differences))
        return dict(comparison=name, endpoint=budget, **stats(seed_means))

    def gap(budget, head, sign=1):
        return [(budget, "r2_" + head, sign), (budget, "r1_" + head, -sign)]

    s2, s3, main_gaps = [], [], []
    for budget in RANGES:
        for mode in MODES[2:]:
            control = "r0_" + mode.split("_")[1]
            s2.append(
                paired_row(
                    mode + " minus " + control, budget, [(budget, mode, 1), (budget, control, -1)]
                )
            )
        for head in ("linear", "mlp"):
            main_gaps.append(paired_row("feature minus early, " + head, budget, gap(budget, head)))
        main_gaps.append(
            paired_row(
                "interaction: MLP gap minus linear gap",
                budget,
                gap(budget, "mlp") + gap(budget, "linear", -1),
            )
        )
    for mode in MODES:
        s3.append(paired_row(mode, "40 minus 12", [(40, mode, 1), (12, mode, -1)]))
    for head in ("linear", "mlp"):
        s3.append(paired_row("fusion gap, " + head, "40 minus 12", gap(40, head) + gap(12, head, -1)))
    # Select checkpoints from each recorded validation history under both limits.
    s4 = []
    histories = defaultdict(dict)
    for row in load_csv(inputs / "training_history.csv"):
        key = (row["mode"], int(row["seed"]), int(row["person"]))
        epoch = int(row["epoch"])
        assert epoch not in histories[key]
        histories[key][epoch] = float(row["inner_mae_bpm"])
    assert len(histories) == 450 and all(
        set(history) == set(range(1, 41)) for history in histories.values()
    )
    selected = {}
    for key, history in histories.items():
        best = float("inf")
        for epoch in range(1, 41):
            if history[epoch] < best - 0.0001:
                best, choice = history[epoch], epoch
            if epoch in RANGES:
                selected[*key, epoch] = choice
    for mode in MODES:
        for budget in RANGES:
            values = [selected[mode, seed, person, budget] for seed in SEEDS for person in range(1, 16)]
            s4.append(
                dict(
                    mode=mode,
                    endpoint=budget,
                    **stats(values),
                    median=median(values),
                    at_limit=values.count(budget),
                )
            )
    same_epoch_count = sum(selected[*key, 12] == selected[*key, 40] for key in histories)
    # Count weights, biases and multiply-accumulates for each encoder and head.
    s6 = []
    config = json.loads((inputs / "architecture.json").read_text())
    for mode in MODES:
        spec = config[mode]
        params = macs = 0
        for channels in spec["branch_input_channels"]:
            for output_channels, kernel_size, output_length in zip(
                spec["widths"], spec["kernels"], spec["output_lengths"]
            ):
                weights = channels * output_channels * kernel_size
                params += weights + output_channels
                macs += weights * output_length
                channels = output_channels
        features = len(spec["branch_input_channels"]) * spec["widths"][-1]
        for width in spec["head_outputs"]:
            params += features * width + width
            macs += features * width
            features = width
        s6.append(dict(mode=mode, parameters=params, conv_linear_macs=macs))
    return {
        "main_six_configs": [row for row in s1 if row["weighting"] == "participant_equal"],
        "main_paired_gaps": main_gaps,
        "S1_full_metrics": s1,
        "S2_same_head": s2,
        "S3_budget": s3,
        "S4_selected_epochs": s4,
        "S4_boundary_counts": [
            dict(
                fits=450,
                selected_at_12=sum(row["at_limit"] for row in s4 if row["endpoint"] == 12),
                selected_at_40=sum(row["at_limit"] for row in s4 if row["endpoint"] == 40),
                same_epoch=same_epoch_count,
            )
        ],
        "S5_activity_summary": s5,
        "S5_activity_by_seed": activity,
        "S6_model_cost": s6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parents[1] / "inputs")
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory for the recomputed CSV tables"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory.")
    tables = calculate(args.inputs)
    args.output.mkdir(parents=True)
    for name, rows in tables.items():
        write_csv(args.output / (name + ".csv"), rows)
    print(f"Wrote {len(tables)} CSV files to {args.output}.")


if __name__ == "__main__":
    main()
