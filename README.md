# The Argmax Gap in Human Chess Move Prediction

Code for reproducing the MAIA3–Allie experiments, tables, and figures. The two base policies are frozen. The trainable components are the linear/MLP/XGBoost selectors, rank-2 correction gate, temperature calibration, and residual refiners.

## Installation

Use Python 3.11 and Git. A CUDA GPU is recommended for full policy inference; the smoke test supports CPU. Full searches and candidate tables require substantially more memory and disk than the smoke test.

```bash
git clone https://github.com/SarveshVGharat/argmax-gap.git
cd argmax-gap
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

`requirements.txt` pins the tested package versions. Install a compatible CUDA build of the pinned PyTorch version if needed for your machine.

## Inputs

The companion **`heldout.parquet`** contains the exact 500,000 development positions in their original order, with only the 22 fields needed by this code (125,119,503 bytes). Its SHA-256 is:

```text
9a00544f74150ef8a5da156464becbccf99ea85a2af4b8f89fef1dc30fef1978
```

This file is necessary for exact reproduction: the original parallel sampling process cannot be recovered from its seed alone. Unused player names, metadata, and duplicate history fields have been removed; retained values and row order are unchanged. Download it from the [v1.0.0 release](https://github.com/SarveshVGharat/argmax-gap/releases/tag/v1.0.0).

Download the companion file, then fetch the official test data, frozen checkpoints, and pinned upstream implementations. The fetch script verifies the companion checksum:

```bash
mkdir -p assets
curl --fail --location \
  https://github.com/SarveshVGharat/argmax-gap/releases/download/v1.0.0/heldout.parquet \
  --output assets/heldout.parquet
python scripts/fetch_assets.py --output-dir assets \
  --heldout-file assets/heldout.parquet
```

Sources and immutable revisions/checksums are defined in `argmax_gap/upstream.py`:

- [MAIA3 implementation](https://github.com/CSSLab/maia3), [79M checkpoint](https://huggingface.co/UofTCSSLab/Maia3-79M).
- [Allie implementation](https://github.com/ippolito-cmu/allie), [medium checkpoint](https://huggingface.co/datasets/yimingzhang/allie-models), [2022 blitz test data](https://huggingface.co/datasets/yimingzhang/allie-data).

The downloads include approximately 4 GB of model weights. Upstream code, weights, and datasets retain their respective terms.

## Run

First run the unit tests and a small end-to-end test:

```bash
python -m unittest discover -s tests -v
python scripts/reproduce.py --assets assets --output runs/smoke --device cpu --smoke
```

The smoke run uses the first 150 positions of each split and reduced nonlinear search grids. Its scores are not paper results. Use a new output directory for each run.

Run the complete experiment:

```bash
python scripts/reproduce.py --assets assets --output runs/paper --device cuda
```

This prepares both datasets, evaluates both policies, fits and selects the downstream methods on development data, evaluates the fixed methods on the test set, generates reports and figures, audits split independence and legal-move alignment, and compares the results with the paper. Full inference and hyperparameter searches are substantially longer than the smoke test.

The command produces:

| Directory/file | Contents |
|---|---|
| `data/` | Aligned test and development positions; construction counts |
| `predictions/` | Full legal-move distributions for each policy and split |
| `methods/models/` | Fitted heads and calibration parameters |
| `methods/selected.json` | Validation-selected configurations |
| `methods/validation_search.csv` | Development search results |
| `methods/predictions/`, `methods/distributions/` | Final method outputs |
| `report/` | Metric/stratum/statistics CSVs and figures |
| `audit.json` | Split, duplicate, and legal-alignment checks |

Candidate feature tables are generated under `methods/candidates/`; they are intermediate data, not release inputs.

## Run individual stages

Every script accepts `--help`. For example:

```bash
python scripts/prepare_data.py --source-format allie-jsonl \
  --input assets/allie-test.jsonl --output data/test.parquet
python scripts/prepare_data.py --source-format heldout-parquet \
  --input assets/heldout.parquet --output data/development.parquet

python scripts/evaluate_policies.py --model maia3 --input data/test.parquet \
  --output predictions/maia3_test.parquet --upstream-root assets/upstream/maia3 \
  --checkpoint assets/maia3-79m.pt --device cuda
python scripts/evaluate_policies.py --model allie --input data/test.parquet \
  --output predictions/allie_test.parquet --upstream-root assets/upstream/allie \
  --checkpoint assets/allie-medium.pt --device cuda
```

Repeat the two inference commands with `data/development.parquet` and output names `maia3_development.parquet` and `allie_development.parquet`. Then:

```bash
python scripts/train_methods.py \
  --heldout-maia predictions/maia3_development.parquet \
  --heldout-allie predictions/allie_development.parquet \
  --test-maia predictions/maia3_test.parquet \
  --test-allie predictions/allie_test.parquet \
  --output runs/methods --diagnostic-time

python scripts/report.py --positions data/test.parquet \
  --maia predictions/maia3_test.parquet --allie predictions/allie_test.parquet \
  --methods runs/methods --output runs/report

python scripts/check_reference.py --report runs/report --require-all
```

Use `--families linear`, `mlp`, `xgboost`, `gates`, `calibration`, `refiner`, or `ensembles` to train selected families. Omit `--methods` from `report.py` for a base-policy report. The comparator checks available methods by default; `--require-all` also fails for missing paper methods. Inference precision and hardware can affect probabilities and close rankings; the reference checker reports discrepancies.

## Paper protocol

- Test construction retains target plies 11 onward, and truncates each game at the first pre-move clock below 30 seconds. Exactly 30 seconds is retained. The result is 884,049 positions from 18,138 games. Clock corrections in the public source are preserved.
- MAIA3 consumes the current board plus up to seven prior boards. Allie consumes the move prefix and prior move-time bits. The realized target move duration is excluded from both policy inputs. The development artifact lacks prior durations; its Allie prefixes use zero durations, matching the original experiment.
- Selector fitting/validation uses a deterministic 400,000/100,000 row split with seed `20260605`. Refiner splitting uses seed `20260606`. These are disjoint row splits; games can occur in both fitting and validation. Test games are independent of all development games.
- Primary selectors use 45 pre-move features. The rank-2 gate adds 20 pre-move features and switches only when its score is **strictly greater** than the validation-selected threshold. `--diagnostic-time` adds the three realized-duration features only to the separately named Table 15 diagnostic variants.
- Standalone calibration uses the original fine temperature search. Refinement starts from its separately fitted coarse temperature calibration. Both use observed move-time buckets and are post-hoc diagnostics. Refiner fitting preserves the original candidate support, clipping, regularization, and residual-scale search.
- Ensemble weights are selected by development NLL. The test alpha sweep is descriptive and does not select a method.
- Top-k uses `1 + count(probability > human_probability)`, preserving ties. The gate preserves base correctness on unswitched rows. NLL/MRR/NDCG/ECE are recomputed from distributions; ECE columns are fractions, Top-k and deltas are percentages/percentage points.
- Position intervals use the paper's paired bootstrap/normal calculations; game intervals resample entire games 10,000 times. Bootstrap endpoints can differ with RNG stream order; reference checks compare point estimates and counts.

## Result map

| Paper results | Generated evidence |
|---|---|
| Tables 1, 5–7; Figure 2 | `metrics.csv`, `topk.pdf` |
| Tables 2–3, 23–24; Figure 3 | `paired_top1.csv`, `complementarity.json`, `complementarity.pdf` |
| Tables 4, 16–17, 19–22; Figures 4, 6, 8 | `strata.csv`, move-time/phase/legal-move PDFs |
| Tables 8–9; Figure 5 | `rank_geometry.csv`, `rank_geometry.pdf` |
| Tables 10–14 | Source feature lists/search grids and `selected.json` |
| Table 15 | Explicitly named duration-diagnostic and pre-move rows in `paired_top1.csv` |
| Table 18; Figure 7 | `ensemble_sweep.csv`, `ensemble_sweep.pdf`; selected ensemble rows in `metrics.csv` |
| Tables 25–28 | `paired_probability_metrics.csv`, `rank_changes.csv`, `time_differences.csv` |
| Appendix F | Data construction summaries, `audit.json`, and `filtering/filtering_sensitivity.csv`; numerical references in `reference/provenance.csv` |
| Figure 1 | `python scripts/figure1.py --output runs/report/figure1.svg` |

For the ensemble curve alone:

```bash
python scripts/ensemble_sweep.py --maia predictions/maia3_test.parquet \
  --allie predictions/allie_test.parquet --output runs/report
```

For the independence/legality audit alone:

```bash
python scripts/audit_splits.py --test data/test.parquet --heldout data/development.parquet \
  --selector-split runs/methods/selector_split.npz \
  --refiner-split runs/methods/refiner_split.npz --output runs/audit.json
```

Add `--test-maia3`, `--test-allie`, `--heldout-maia3`, and `--heldout-allie` to validate the corresponding prediction files too.

For Table 30's source-versus-retained filtering sensitivity:

```bash
python scripts/filtering_sensitivity.py --source-jsonl assets/allie-test.jsonl \
  --retained data/test.parquet --output runs/report/filtering \
  --reference reference/provenance.csv
```
