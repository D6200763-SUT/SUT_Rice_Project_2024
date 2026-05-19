# CNN-LSTM Weighted Loss  (spike_weight=8.0)

spike_threshold (log1p): 1.0986
Train spikes: 10521/42228 (24.9%)

## Test Metrics (log1p)
- R²:   0.1104  (WORSE vs baseline 0.484)
- MAE:  1.1089
- RMSE: 1.5732
- sMAPE:1.5683

## Test Metrics (raw)
- R²:   -0.0036  (WORSE vs baseline 0.0041)
- MAE:  30.96
- RMSE: 220.36  (WORSE vs baseline 219.51)

## Spike-only Test Metrics (y_true > threshold)
- R²:   -0.0671
- RMSE: 1.5657
- MAE:  1.1974
- n:    1780