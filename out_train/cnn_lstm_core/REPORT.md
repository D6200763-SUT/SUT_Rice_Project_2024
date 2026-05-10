# Training report: CNN-LSTM
NPZ: D:\AI_Projects\SUT_Rice_Project_2024\out_feature_sets\core\sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Metrics (log1p scale)
- TEST MAE:  0.9340
- TEST RMSE: 1.4790
- TEST R2:   0.2098
- TEST sMAPE:1.7274

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.50
- TEST RMSE: 218.63
- TEST R2:   -0.0119
- TEST sMAPE:1.8678

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)