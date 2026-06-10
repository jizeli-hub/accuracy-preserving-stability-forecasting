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

1. Accept the competition rules on Kaggle.
2. Install and configure the Kaggle CLI.
3. Run:

```bash
bash scripts/download_m5_data.sh
```

Raw files are excluded from Git because several exceed GitHub's normal
per-file size limit and redistribution is governed by the official competition
terms. Accept the Kaggle competition rules before downloading and using them.
