# Two-stage BPH Model — Results

## Configuration
- NPZ: results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz
- Spike quantile: Q75% = 1.0986 (raw ≈ 2.0 ตัว)
- Spike ratio (train): 28.1%
- Spike threshold (inference): 0.5

## Stage 1: Classifier
| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.7816 |
| Precision  | 0.5335 |
| Recall     | 0.8662 |
| F1         | 0.6603 |
| AUC-ROC    | 0.8759 |
| TP / FP / FN / TN | 1768 / 1546 / 273 / 4743 |

## Stage 2 + Combined: Regression (Test Set)
| Metric       | log1p  | raw BPH |
|-------------|--------|---------|
| R²          | 0.2093 | 0.0286 |
| MAE         | 0.7910 | 28.7375 |
| RMSE        | 1.4794 | 214.2162 |
| SMAPE       | 0.5196 | 0.6399 |

## เปรียบเทียบกับ Single-stage
| | Single-stage CNN-LSTM | Two-stage |
|---|---|---|
| R² (log1p) | N/A | 0.2093 |
| Delta R²   | — | **N/A** |

## Spike-only Performance (top 25%)
| Metric | log1p | raw |
|--------|-------|-----|
| R²     | 0.0627 | -0.0048 |
| RMSE   | 1.5925 | 429.2380 |