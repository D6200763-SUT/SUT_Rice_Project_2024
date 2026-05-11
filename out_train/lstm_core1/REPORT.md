# Training report: LSTM
NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/out_feature_sets/core/sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Split counts (samples)
- train: 42432
- val:   8262
- test:  8330

## Metrics (log1p scale)
- TEST MAE:  0.9310
- TEST RMSE: 1.4750
- TEST R2:   0.2140
- TEST sMAPE:1.7115

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.33
- TEST RMSE: 218.39
- TEST R2:   -0.0097
- TEST sMAPE:1.8479

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)