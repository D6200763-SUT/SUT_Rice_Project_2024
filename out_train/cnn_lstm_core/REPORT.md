# Training report: CNN-LSTM
NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/out_feature_sets/core/sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Metrics (log1p scale)
- TEST MAE:  0.9394
- TEST RMSE: 1.4863
- TEST R2:   0.2020
- TEST sMAPE:1.7285

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.52
- TEST RMSE: 218.61
- TEST R2:   -0.0117
- TEST sMAPE:1.8686

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)