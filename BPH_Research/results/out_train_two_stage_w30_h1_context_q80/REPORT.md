# Two-stage BPH Model — Results

## Configuration
- NPZ: results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz
- Spike quantile: Q80% = 1.9459 (raw ≈ 6.0 ตัว)
- Spike ratio (train): 20.2%
- Spike threshold (inference): 0.5

## Stage 1: Classifier
| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.7714 |
| Precision  | 0.4386 |
| Recall     | 0.9383 |
| F1         | 0.5978 |
| AUC-ROC    | 0.9130 |
| TP / FP / FN / TN | 1415 / 1811 / 93 / 5011 |

## Stage 2 + Combined: Regression (Test Set)
| Metric       | log1p  | raw BPH |
|-------------|--------|---------|
| R²          | 0.0343 | 0.0239 |
| MAE         | 0.8982 | 29.8618 |
| RMSE        | 1.6349 | 214.7316 |
| SMAPE       | 0.5311 | 0.6537 |

## เปรียบเทียบกับ Single-stage
| | Single-stage CNN-LSTM | Two-stage |
|---|---|---|
| R² (log1p) | N/A | 0.0343 |
| Delta R²   | — | **N/A** |

## Spike-only Performance (top 20%)
| Metric | log1p | raw |
|--------|-------|-----|
| R²     | -0.2004 | -0.0305 |
| RMSE   | 1.4084 | 499.9144 |