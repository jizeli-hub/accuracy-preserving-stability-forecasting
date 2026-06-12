# Runtime Source Records

These files are the per-run source observations for the runtime scalability
table in the paper.

Each scale contains three runs with seeds 42, 123, and 2024. The main
configuration uses full selected XGBoost training rows and 300 trees. The JSON
files preserve the command-line configuration resolved by the experiment
runner.

The files aggregate to:

```text
../large_scale_runtime_results.csv
```

The rounded values printed in the manuscript are in:

```text
../table_runtime_scalability.csv
```

Verify both with:

```bash
python scripts/verify_runtime_table.py
```
