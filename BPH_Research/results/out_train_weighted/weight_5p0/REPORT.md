# CNN-LSTM Weighted Loss  (spike_weight=5.0)

spike_threshold (log1p): 1.0986
Train spikes: 10521/42228 (24.9%)

## Test Metrics (log1p)
- R²:   0.2664  (WORSE vs baseline 0.484)
- MAE:  0.9292
- RMSE: 1.4287
- sMAPE:1.5828

## Test Metrics (raw)
- R²:   -0.0043  (WORSE vs baseline 0.0041)
- MAE:  30.15
- RMSE: 220.43  (WORSE vs baseline 219.51)

## Spike-only Test Metrics (y_true > threshold)
- R²:   -0.2992
- RMSE: 1.7276
- MAE:  1.3162
- n:    1780