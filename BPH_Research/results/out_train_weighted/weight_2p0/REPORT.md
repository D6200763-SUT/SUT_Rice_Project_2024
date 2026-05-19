# CNN-LSTM Weighted Loss  (spike_weight=2.0)

spike_threshold (log1p): 1.0986
Train spikes: 10521/42228 (24.9%)

## Test Metrics (log1p)
- R²:   0.3171  (WORSE vs baseline 0.484)
- MAE:  0.8261
- RMSE: 1.3784
- sMAPE:1.6250

## Test Metrics (raw)
- R²:   0.0011  (WORSE vs baseline 0.0041)
- MAE:  26.71
- RMSE: 219.84  (WORSE vs baseline 219.51)

## Spike-only Test Metrics (y_true > threshold)
- R²:   -0.9703
- RMSE: 2.1275
- MAE:  1.6041
- n:    1780