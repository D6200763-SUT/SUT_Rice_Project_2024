# Training report: LSTM
NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/out_feature_sets_nan_safe/core/sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Split counts (samples)
- train: 42432
- val:   8262
- test:  8330

## Metrics (log1p scale)
- TEST MAE:  0.9340
- TEST RMSE: 1.4827
- TEST R2:   0.2058
- TEST sMAPE:1.7148

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.36
- TEST RMSE: 218.45
- TEST R2:   -0.0102
- TEST sMAPE:1.8502

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)