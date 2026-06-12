# Accuracy-Preserving Stability Regularization for Retail Demand Forecasting

Research code and reproducibility artifacts for large-scale M5 retail demand forecasting with:

- XGBoost structured-feature baselines
- a lightweight TimeMixer-style temporal encoder
- hybrid temporal and structured feature fusion
- training-time forecast stability regularization
- post-hoc smoothing baselines
- accuracy, stability, robustness, runtime, and leakage evaluation

The main experiments use 1,000, 3,000, and 4,000 M5 item-store demand series. The selected stability-aware setting uses `lambda = 0.05`.

## Repository Layout

```text
paper/       Anonymous ICEME manuscript and bibliography
src/         Data processing, models, evaluation, and utilities
results/     Selected experiment tables and audit outputs
figures/     Main publication figures
data/        Local dataset directory and download instructions
notebooks/   Optional exploratory notebooks
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The code expects the official M5 Forecasting Accuracy files under:

```text
data/m5-forecasting-accuracy/
```

See [data/README.md](data/README.md) for official Kaggle download instructions.
Raw competition data is not redistributed in this public repository.

The checksums in `data/M5_SHA256SUMS.txt` identify the exact M5 files used for
the reported experiments.

## Baseline and Hybrid Experiment

```bash
python -m src.models.xgboost_baseline_pipeline \
  --data-dir data/m5-forecasting-accuracy \
  --max-series 500
```

## Large-Scale Robustness Experiment

Example full-configuration run:

```bash
python -m src.models.large_scale_experiments \
  --data-dir data/m5-forecasting-accuracy \
  --max-series 1000 3000 4000 \
  --random-seed 42 123 2024 \
  --num-runs 3 \
  --stability-lambda 0.05 \
  --xgboost-n-estimators 300 \
  --output-dir .
```

Large experiments can require substantial memory and runtime. Start with a smaller `--max-series` value for a smoke test.

## Reproducing the Reported Tables

The exact three-seed runtime observations behind the paper's runtime table are
committed under `results/runtime_sources/`. Recompute and verify the aggregate
table with:

```bash
python scripts/verify_runtime_table.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full experiment commands,
output provenance, and expected hardware-dependent differences.

## Paper

The anonymous submission manuscript is available in:

```text
paper/Accuracy-Preserving-Stability-Regularization-ICEME-2026.docx
```

## Data Leakage Controls

The pipeline uses chronological train-validation splits, shifted lag and rolling features, training-only fitted encoders, and same-series training prediction pairs for the stability objective. The selected audit summary is included in `results/`.

## Notes

- The TimeMixer-style component is a lightweight temporal feature extractor, not a full end-to-end reproduction of the original TimeMixer architecture.
- The capped 5,000-series diagnostic is not used for the main scalability or runtime claims.
- Raw M5 files are subject to the official Kaggle competition terms.
