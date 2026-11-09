# Provenance and validation

The archived manuscript inputs describe the completed 450-trajectory study. Earlier fixed-split, spectral and 180-trajectory studies, partial executions, cloud deployment records, meeting notes, manuscript drafts and literature PDFs are excluded from this independent repository.

## Portable adaptation

`src/study.py` retains the original `model_spec`, `Encoder`, `Model`, `build_model`, `nominal_macs`, `selection_mae` and `select_checkpoints` function/class bodies. Constants are read from the supplied configuration and split files. `src/data.py` retains the original participant validation, scaler validation, `Pool` and `Dataset` bodies and binds them to those constants. `scripts/train.py` replaces the original cloud-budget and deployment orchestration with a local Linux CPU runner; it preserves initialization, optimizer, per-epoch sample ordering, float32 validation scoring, fixed 40-epoch trajectories and the full-grid outer-evaluation gate.

`prepare_data.py` combines only the wrist-signal portions of the original extraction, alignment and CNN-cache stages. Native wrist arrays are stored as float32, ACC is resampled in float64 and stored as float32, and both signal windows are processed in float64 before final float32 cache storage. It preserves participant/window ordering, label values, per-window filtering and demeaning. Activity labels use zero-order holding from 4 to 64 Hz and the most frequent class in each 8-second window, with the lowest code resolving ties.

The plotting functions retain the manuscript's graphical encodings but read compact, independent inputs. The two full group-level Figure 5 prediction series allow the disclosed candidate enumeration to be replayed. A fallback font supports Linux systems without Times New Roman.

The original implementations are identified by `provenance/original_sources.json`. These hashes identify source versions; adapted files have different hashes. The original training protocol SHA-256 is `bfc78984f4dd4cd7e25158439b267161475be0e71972cff96109e3594d20aee5`. Infrastructure identities, credentials, hostnames and original machine-specific paths are excluded.

## Packaging validation

- `preparation_validation.json`: all 15 official participant recordings were processed into 64,697 windows; all 12 cache files match the historical SHA-256 checksums.
- `adapter_validation.json`: all six original/adapted model initial states, synthetic forward outputs and one synthetic AdamW update are identical in the checked environment; all 15 training-only scalers match the original implementation. This is a software-parity check, not a new participant experiment.
- The standalone reproduction command regenerates the two main tables, all six figures and the full numerical summaries without opening the original research checkout. Overall and activity summaries match the archived numeric results within absolute tolerance `1e-10`.
- Five portable synthetic unit tests cover parameter/MAC counts, matched encoder initialization, checkpoint ties/ranges, float32 clipping, signal preprocessing and split integrity.

`provenance/input_sha256.json` checks the packaged analysis inputs before regeneration. The selected-epoch diagnostic tool replays the strict selection rule over all 18,000 exported history rows and checks all 900 selections. Exported CSV validation is distinct from verifying the original private training records; the output audit preserves that distinction.

No new participant training or participant model forward pass was performed during packaging. Historical model weights are not included. The new runner is tested through synthetic parity and inspected against the original logic; an independent 450-trajectory rerun would be needed to assess exact training reproduction on another environment.
