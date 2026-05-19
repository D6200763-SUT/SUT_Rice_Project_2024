# Stage 2: BPH Two-stage Regression — Report

NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/BPH_Research/results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz
Window=30, Features=18
Spike threshold (log1p): 1.0986
Stage 1 pred_threshold:  0.5

## Stage 2 Architecture: CNN-LSTM Regression (Optuna best params)
- Conv1D(128, k=5) + MaxPool(2) + Dropout(0.1094)
- LSTM(32, dropout=0.1094)
- Dense(64, relu) + Dense(1)  — Huber loss
- lr=0.000433, clipnorm=2.0, batch_size=256

## Spike Sample Counts (Stage 1 predicted)
- Train: 14724/42228 (34.9%)
- Val:   3276/8058 (40.7%)
- Test:  2872/8126 (35.3%)

## Two-stage Inference Logic
- Stage 1 = 0 (no-spike) → y_pred_final = 0.0
- Stage 1 = 1 (spike)    → y_pred_final = Stage 2 regression output

## Spike-only Metrics (Stage 2 on spike test samples)
- R²  log1p: 0.0919
- MAE log1p: 1.5679
- n_samples: 2872

## Full Two-stage Metrics (all test samples)
  R²  log1p: 0.3109  (baseline 0.4798) [WORSE Δ=-0.1689]
  MAE log1p:  0.7014
  RMSE log1p: 1.3847
  R²  raw: 0.0063  (baseline 0.0041) [BETTER Δ=+0.0022]
  MAE raw:    25.65
  RMSE raw: 219.2676  (baseline 219.51) [BETTER Δ=+0.2424]

## Next Steps
- If two-stage R² > 0.4798: include in paper as main result
- Consider optimizing pred_threshold (try 0.3–0.4 for higher recall)
- Consider training Stage 2 on oracle spike labels (y_true > threshold) for comparison