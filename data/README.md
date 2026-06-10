# M5 Dataset

This project uses the official **M5 Forecasting - Accuracy** competition data.

Expected files:

```text
data/m5-forecasting-accuracy/
  calendar.csv
  sales_train_evaluation.csv
  sales_train_validation.csv
  sample_submission.csv
  sell_prices.csv
```

## Included Compressed Files

The repository includes gzip-compressed CSV files in:

```text
data/m5-forecasting-accuracy-gzip/
```

After confirming compliance with the official competition terms, decompress
them for local use:

```bash
mkdir -p data/m5-forecasting-accuracy
gzip -dk data/m5-forecasting-accuracy-gzip/*.csv.gz
mv data/m5-forecasting-accuracy-gzip/*.csv data/m5-forecasting-accuracy/
```

`SHA256SUMS.txt` records checksums for the compressed archives.

## Download with the Kaggle CLI

1. Accept the competition rules on Kaggle.
2. Install and configure the Kaggle CLI.
3. Run:

```bash
bash scripts/download_m5_data.sh
```

The uncompressed CSV files are excluded from Git because several exceed
GitHub's normal per-file size limit. Verify the official competition terms
before redistributing or using the dataset.
