# Two-stage BPH Model — Results

## Configuration
- NPZ: results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz
- Spike quantile: Q75% = 1.0986 (raw ≈ 2.0 ตัว)
- Spike ratio (train): 28.1%
- Spike threshold (inference): 0.5

## Stage 1: Classifier
| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.7729 |
| Precision  | 0.5225 |
| Recall     | 0.8491 |
| F1         | 0.6469 |
| AUC-ROC    | 0.8773 |
| TP / FP / FN / TN | 1733 / 1584 / 308 / 4705 |

## Stage 2 + Combined: Regression (Test Set)
| Metric       | log1p  | raw BPH |
|-------------|--------|---------|
| R²          | 0.1449 | 0.0014 |
| MAE         | 0.8269 | 30.0913 |
| RMSE        | 1.5385 | 217.1912 |
| SMAPE       | 0.5399 | 0.6563 |

## เปรียบเทียบกับ Single-stage
| | Single-stage CNN-LSTM | Two-stage |
|---|---|---|
| R² (log1p) | N/A | 0.1449 |
| Delta R²   | — | **N/A** |

## Spike-only Performance (top 25%)
| Metric | log1p | raw |
|--------|-------|-----|
| R²     | 0.0673 | -0.0240 |
| RMSE   | 1.5885 | 433.3351 |