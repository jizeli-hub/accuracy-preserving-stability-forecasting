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
data/        Compressed M5 files and download instructions
notebooks/   Optional exploratory notebooks
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Compressed copies of the M5 Forecasting Accuracy CSV files are included under:

```text
data/m5-forecasting-accuracy-gzip/
```

After confirming that your use complies with the official competition terms,
decompress them into the directory expected by the code:

```text
data/m5-forecasting-accuracy/
```

See [data/README.md](data/README.md) for decompression and official Kaggle download instructions.

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
- M5 files are subject to the official Kaggle competition terms.
