# CNN-LSTM Weighted Loss  (spike_weight=3.0)

spike_threshold (log1p): 1.0986
Train spikes: 10521/42228 (24.9%)

## Test Metrics (log1p)
- R²:   0.3442  (WORSE vs baseline 0.484)
- MAE:  0.8278
- RMSE: 1.3508
- sMAPE:1.5926

## Test Metrics (raw)
- R²:   0.0079  (BETTER vs baseline 0.0041)
- MAE:  27.00
- RMSE: 219.09  (BETTER vs baseline 219.51)

## Spike-only Test Metrics (y_true > threshold)
- R²:   -0.3405
- RMSE: 1.7549
- MAE:  1.2963
- n:    1780