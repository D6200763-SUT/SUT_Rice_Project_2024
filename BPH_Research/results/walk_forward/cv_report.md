## Walk-Forward Cross-Validation Results

| Metric              | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |
|---------------------|--------|--------|--------|-----------|
| Stage1 Recall       | 0.743 | 0.899 | 0.779 | 0.807±0.067 |
| Stage1 Precision    | 0.250 | 0.333 | 0.432 | 0.339±0.074 |
| Stage1 F1           | 0.374 | 0.486 | 0.556 | 0.472±0.075 |
| Stage1 AUC          | 0.804 | 0.915 | 0.923 | 0.881±0.054 |
| Best Threshold      | 0.300 | 0.300 | 0.500 | 0.367±0.094 |
| Stage2 R² (log1p)   | -0.130 | -0.004 | -0.223 | -0.119±0.090 |
| Stage2 RMSE_log1p   | 0.847 | 0.739 | 0.662 | 0.749±0.076 |
| Stage2 R² (raw)     | -0.005 | 0.024 | -0.003 | 0.006±0.013 |
| Stage2 RMSE_raw     | 1199.505 | 965.010 | 618.308 | 927.608±238.742 |

**Note:** Fold 4 test period (2019) corresponds to the original single-split evaluation.
Best threshold selected per fold to achieve Recall ≥ 0.80 on the validation set.

**Baseline (single-stage CNN-LSTM, Fold 4 test period):** R²_log1p = 0.484