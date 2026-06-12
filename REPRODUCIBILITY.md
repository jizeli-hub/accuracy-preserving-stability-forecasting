# Reproducibility Guide

This guide maps the public repository artifacts to the experiments reported in
the paper. The main results use 1,000, 3,000, and 4,000 item-store series,
seeds 42, 123, and 2024, a 28-day validation horizon, 300 XGBoost trees, and
stability weight `lambda = 0.05`.

## 1. Obtain the M5 Data

Follow `data/README.md`. Raw M5 competition files are not redistributed by this
repository. They must be downloaded from Kaggle after accepting the competition
rules.

## 2. Create the Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Smoke Test

```bash
python -m src.models.large_scale_experiments \
  --data-dir data/m5-forecasting-accuracy \
  --max-series 10 \
  --random-seed 42 \
  --num-runs 1 \
  --stability-lambda 0.05 \
  --xgboost-n-estimators 10 \
  --output-dir reproduction_smoke
```

## 4. Main Three-Seed Experiments

The reported scales were run separately with the same full-row, 300-tree
configuration:

```bash
for scale in 1000 3000 4000; do
  python -m src.models.large_scale_experiments \
    --data-dir data/m5-forecasting-accuracy \
    --max-series "${scale}" \
    --random-seed 42 123 2024 \
    --num-runs 3 \
    --validation-days 28 \
    --stability-lambda 0.05 \
    --timemixer-history-length 28 \
    --timemixer-embedding-dim 16 \
    --timemixer-max-iter 80 \
    --timemixer-max-train-rows 50000 \
    --xgboost-n-estimators 300 \
    --output-dir "reproduction_runs/series_${scale}"
done
```

No XGBoost training-row cap is used for the 1,000-, 3,000-, or 4,000-series
main experiments.

## 5. Runtime Table Provenance

The paper's runtime table is derived from:

```text
results/runtime_sources/series_1000_runtime_per_run.csv
results/runtime_sources/series_3000_runtime_per_run.csv
results/runtime_sources/series_4000_runtime_per_run.csv
```

The corresponding configurations are stored in the three metadata JSON files
in the same directory. Their aggregation is committed as
`results/large_scale_runtime_results.csv`, and the rounded paper table is
`results/table_runtime_scalability.csv`.

Run the audit:

```bash
python scripts/verify_runtime_table.py
```

The audit recomputes means, sample standard deviations, maxima, and run counts
from the per-run observations and checks the rounded publication table.

Wall-clock runtime and peak resident memory depend on hardware, operating
system, background load, and library versions. A reproduction should preserve
the configuration and qualitative scaling pattern; identical absolute seconds
are not expected on a different machine.

## 6. Data Scope

The main paper does not use the capped 5,000-series diagnostic as primary
evidence. Promotion/event-window error decomposition and inventory-cost
simulation are not reported experiments and are left as future work; they are
not required to reproduce the current paper tables.
