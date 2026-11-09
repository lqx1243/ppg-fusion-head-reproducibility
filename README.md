# Regression heads and PPG ACC fusion reproducibility

This repository supports **Joint Evaluation of Regression Heads and PPG–ACC Fusion Strategies for Wrist-Based Heart Rate Estimation** by Qixin Luo. It provides the six model configurations, fixed participant splits, training and preprocessing code, archived participant-by-seed results, and commands to regenerate manuscript tables and figures.

The study is an empirical comparison of fusion rankings under different regression heads and checkpoint-selection budgets. Feature fusion improves on early fusion with an MLP within epochs 1–40, while PPG-only with an MLP has the lowest overall mean MAE. The results do not demonstrate a general benefit from adding ACC. The 15-participant cohort had been explored previously; this is an internal participant-held-out evaluation.

**Availability:** prepared as an independent local repository on 9 November 2026. No hosted repository URL or collaborator access has been established by this package. Upload and access arrangements are separate steps. Contact: qixin.luo.contact@gmail.com.

## Regenerate the archived manuscript results

Requires Python 3.11 or newer. The following command only analyzes supplied results; it does not train models or download recordings.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-analysis.txt
python scripts/reproduce.py
```

The generated files are under `outputs/`:

- `tables/overall_table.tex` and `tables/activity_table.tex`: the two main result tables, with matching Chinese Markdown tables and full numerical CSV summaries.
- `figures/signals.pdf`, `fusion_routes.pdf`, `interaction.pdf`, `participants.pdf`, and `cases.pdf`: manuscript Figures 1–5.
- `figures/training_validation_curves.pdf` and `training/training_epochs_table.tex`: the added validation-curve figure and checkpoint-epoch table.
- `tables/S1_full_metrics.csv`, `S2_same_head.csv`, `S3_budget.csv`, and `S5_activity_summary.csv`: complete overall, matched-head ACC, selection-budget, and activity summaries.

`python scripts/reproduce.py --tables-only` regenerates numerical tables without plotting. The results use participant-equal means and population standard deviations across the five seeds; window-weighted summaries are also retained. A seed is not an independent participant.

Times New Roman is used when installed; otherwise figures use DejaVu Serif. This affects typography, not results. No conference template is included because the target conference is not final.

## Included evidence

| File | Content |
|---|---|
| `inputs/participant_by_seed.csv` | 900 rows: 15 participants × 5 seeds × 6 configurations × 2 selection ranges |
| `inputs/person_activity_metrics.csv` | 8,100 rows, including explicit missing S6 lunch, walking and working cells |
| `inputs/training_history.csv` | All 18,000 epoch records from the 450 completed trajectories |
| `inputs/training_selected_epochs.csv` | All 900 selected epochs and selection scores |
| `inputs/case_walking_predictions.csv` | Every walking window in the two previously selected Figure 5 participant/activity groups |
| `inputs/case_candidates.csv` | All 435 eligible 120-second candidates |
| `inputs/case_selected_predictions.csv` | The 114 displayed window predictions |
| `inputs/signal_example.npz` | Only the signal segments needed for Figures 1 and 2 |
| `configs/study.json` | Model, optimizer, seed, endpoint and selection definitions |
| `splits/loso.json` | All 15 fixed 11-train/3-validation/1-held-out assignments |
| `splits/storage_partitions.json` | Historical storage ordering, separate from the LOSO role assignment |

The five seeds are 3001–3005. Each trajectory runs for 40 epochs, with checkpoints selected within epochs 1–12 and 1–40 of the same trajectory. Selection uses pooled inner-validation MAE in float32, clipped to 30–220 bpm, with strict improvement greater than 0.0001 bpm. Normalization is fitted only on the 11 training participants. There is no early stopping, schedule, augmentation, seed selection, or refitting on 14 participants.

Figure 5 is a post-hoc illustration of selected behaviors. The script re-enumerates all consecutive 57-window candidates in S8 and S14 walking and verifies the disclosed choices: minimize R0-MLP segment error for S8 and R2-MLP segment error for S14, with earliest start breaking ties. The union of 57 eight-second windows, stepped every two seconds, covers 120 seconds. Overall conclusions use the complete participant and activity results.

## Prepare recordings for a new run

Obtain PPG-DaLiA from the [official UCI dataset page](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia), DOI [10.24432/C53890](https://doi.org/10.24432/C53890). Extract the recordings so that the chosen directory contains `S1/S1.pkl` through `S15/S15.pkl`. The full recordings and trained weights are not redistributed here. See [data attribution and rights](DATA_AND_RIGHTS.md).

Use an environment with NumPy, SciPy and CPU PyTorch. For the historical Linux CPU package pins, see `environment/requirements-linux-cpu.lock.txt`; its original platform and Python version are recorded in `environment/observed_runtime.json`. A contemporary environment can run the analysis, but does not establish bitwise reproduction of training.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r environment/requirements-training.txt
python scripts/prepare_data.py --raw /path/to/PPG_FieldStudy --output work/preparation
```

The preprocessing command refuses an existing output directory. It creates `work/preparation/cache/`, `activities.csv`, and a preparation report. It extracts native wrist PPG and ACC, applies the original float32 storage boundaries, resamples ACC to 64 Hz, and forms the same 8-second/2-second-step windows. It bandpass-filters and centers PPG per window and centers each ACC axis. Supplied HR labels are preserved. All 12 generated cache files must match the historical SHA-256 manifest before training proceeds; mismatches are retained for investigation.

This preparation route was run on all 15 original recordings during packaging: every cache file matched the historical training inputs byte for byte. The report is in `provenance/preparation_validation.json`.

## Run the complete study again

A new run requires Linux, CPU-only PyTorch, enough local disk for the input cache, checkpoints and prediction vectors, and time for 450 training trajectories. The archived study took approximately 27 hours on an 8-vCPU Intel Xeon CPU; this is context, not a runtime guarantee. The portable command uses eight intra-op threads and one inter-op thread.

```sh
python -m unittest discover -s tests -v
python scripts/train.py --cache work/preparation/cache --run work/new_run
```

The runner refuses an existing run directory. A failed run and its completed epoch records remain available for inspection; it does not automatically retry or resume. All 450 trajectories and both selected checkpoints are checked and frozen before any selected outer-participant model inference. The runner then saves all 2,700 train/inner/outer prediction vectors and the 900 participant-by-seed endpoint results. It never starts a cloud instance or uses cloud credentials.

The portable orchestration is adapted from the archived implementation; the model, data-view and checkpoint-selector function bodies are retained. Original source fingerprints and adaptation details are in [provenance and validation](docs/PROVENANCE.md). No new participant training was performed while assembling this repository, and the portable runner has not been certified to reproduce historical weights bit for bit.

## Analyze a newly trained run

Keep new outputs separate from the archived manuscript inputs. Export the completed run, then use its files explicitly:

```sh
python scripts/export_results.py --run work/new_run --activities work/preparation/activities.csv --output work/new_inputs
python src/recompute_tables.py --inputs work/new_inputs --output work/new_tables
python scripts/training_diagnostics.py --history-csv work/new_inputs/training_history.csv --selected-csv work/new_inputs/training_selected_epochs.csv --output-dir work/new_diagnostics --figure-dir work/new_figures
```

The export preserves all participant/activity cells and provides the bridge from saved predictions to numerical summaries. `scripts/reproduce.py` always regenerates the **archived manuscript** outputs. Figures 1–2 and the selected Figure 5 examples retain their disclosed original inputs; they are not silently replaced by a new run.
