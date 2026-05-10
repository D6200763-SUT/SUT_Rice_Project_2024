# Training report: CNN-LSTM
NPZ: D:\AI_Projects\SUT_Rice_Project_2024\out_feature_sets\core\sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Metrics (log1p scale)
- TEST MAE:  0.9174
- TEST RMSE: 1.4909
- TEST R2:   0.1970
- TEST sMAPE:1.7362

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.48
- TEST RMSE: 218.67
- TEST R2:   -0.0122
- TEST sMAPE:1.8713

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)