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

## Download with the Kaggle CLI

1. Sign in to Kaggle and accept the rules for the
   [M5 Forecasting - Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
2. Install the Kaggle CLI:

```bash
python -m pip install kaggle
```

3. Create an API token from the Kaggle account settings page, place
   `kaggle.json` under `~/.kaggle/`, and restrict its permissions:

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

4. Run:

```bash
bash scripts/download_m5_data.sh
```

5. Verify the downloaded files:

```bash
shasum -a 256 -c data/M5_SHA256SUMS.txt
```

Raw files are excluded from Git because several exceed GitHub's normal
per-file size limit and redistribution is governed by the official competition
terms. Accept the Kaggle competition rules before downloading and using them.

The checksum manifest records the exact official files used for the reported
experiments. If Kaggle republishes a file, investigate any checksum difference
before attempting an exact reproduction.
