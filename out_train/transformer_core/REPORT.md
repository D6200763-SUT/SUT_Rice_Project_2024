# Training report: Transformer
NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/out_feature_sets/core/sequences_window30_h1.npz
Window=30, Horizon=1, Features=14

## Metrics (log1p scale)
- TEST MAE:  0.9363
- TEST RMSE: 1.5041
- TEST R2:   0.1826
- TEST sMAPE:1.7240

## Metrics (raw scale = expm1(log1p))
- TEST MAE:  27.58
- TEST RMSE: 218.44
- TEST R2:   -0.0101
- TEST sMAPE:1.8609

## Saved files
- metrics.json
- predictions_test.csv
- figures/loss_curve.png
- figures/scatter_true_pred_log1p.png
- figures/residual_hist_log1p.png
- figures/timeseries_test_Chachoengsao Rice research Center.png (if available)