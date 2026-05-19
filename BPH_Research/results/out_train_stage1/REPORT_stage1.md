# Stage 1: BPH Spike Binary Classifier

NPZ: /home/ai-station/Projects_code/SUT_Rice_Project_2024/BPH_Research/results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz
Window=30, Features=18
Spike threshold (Q75% of y_train, log1p): 1.0986
Class weight: {0: 1.00, 1: 3.01}

## Architecture: CNN-LSTM Binary Classifier
- Conv1D(64, k=5) + BatchNorm + MaxPool(2) + Dropout(0.25)
- Conv1D(32, k=5) + BatchNorm + Dropout(0.25)
- LSTM(64, dropout=0.25)
- Dense(32, relu) + Dropout(0.125) + Dense(1, sigmoid)

Rationale: CNN-LSTM is the best model on this dataset (R²=0.484 W30 H7).
CNN blocks capture local temporal patterns preceding spikes.
BatchNorm stabilizes training under class imbalance.

## Metrics
### Train
- Accuracy:  0.8333
- Precision: 0.6182
- Recall:    0.8652
- F1:        0.7212
- Samples:   pos=10521  neg=31707
- Confusion matrix: [[26086, 5621], [1418, 9103]]

### Val
- Accuracy:  0.7561
- Precision: 0.5040
- Recall:    0.8292
- F1:        0.6269
- Samples:   pos=1991  neg=6067
- Confusion matrix: [[4442, 1625], [340, 1651]]

### Test
- Accuracy:  0.7797
- Precision: 0.4983
- Recall:    0.8039
- F1:        0.6152
- Samples:   pos=1780  neg=6346
- Confusion matrix: [[4905, 1441], [349, 1431]]

## Thresholds
- Spike threshold saved: /home/ai-station/Projects_code/SUT_Rice_Project_2024/BPH_Research/results/out_train_stage1/threshold.json
- Binary decision threshold (pred_threshold): 0.5

## Next Steps
- F1 >= 0.55 and Recall >= 0.60 → proceed to Stage 2 regression
- Otherwise: tune --quantile (try 0.70) or increase --patience