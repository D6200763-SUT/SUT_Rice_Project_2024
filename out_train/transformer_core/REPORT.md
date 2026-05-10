# Training report: Transformer
NPZ: D:\AI_Projects\SUT_Rice_Project_2024\out_feature_sets\core\sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Metrics (log1p scale)
- TEST MAE:  0.9135
- TEST RMSE: 1.5029
- TEST R2:   0.1840
- TEST sMAPE:1.7161

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.59
- TEST RMSE: 218.14
- TEST R2:   -0.0073
- TEST sMAPE:1.8522

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)