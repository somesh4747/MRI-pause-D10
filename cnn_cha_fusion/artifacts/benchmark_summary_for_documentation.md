# Benchmark Summary (CNN + CHA Fusion v2)

## 1) Model Comparison
We compared the Fusion CNN+CHA model with three baseline classifiers trained on CHA voice features:
- Logistic Regression
- Random Forest
- SVM (RBF)

Best-performing model by Macro-F1: **Voice RandomForest**

Fusion model core metrics:
- Accuracy: 0.7576
- Precision (MCI): 0.7805
- Recall (MCI): 0.7218
- F1-score (MCI): 0.7500
- Macro-F1: 0.7574
- Dice coefficient (MCI): 0.7500
- Balanced Accuracy: 0.7578

## 2) Correlation Matrix
The CHA feature correlation matrix highlights redundancy and complementary speech markers.
Features with strongest absolute correlation with target are listed in notebook output and support class-separability analysis.

## 3) Dice Hit-Map
The Dice hit-map summarizes class overlap quality per model for:
- Control class
- MCI class

Higher Dice indicates better overlap between predicted and true regions for the class.

## 4) Error Type Analysis
Fusion confusion details:
- TN: 208
- FP: 54
- FN: 74
- TP: 192

Interpretation:
- FP are Control samples incorrectly flagged as MCI.
- FN are MCI samples missed as Control.
- Confidence and distance-to-threshold plots reveal whether errors are mostly boundary cases or high-confidence failures.

## 5) Generated Documentation Assets
- benchmark_model_comparison.csv
- benchmark_model_metric_comparison.png
- benchmark_cha_correlation_matrix.png
- benchmark_dice_hit_map.png
- benchmark_error_type_distribution.png
- benchmark_error_confidence_boxplot.png
- benchmark_fusion_error_analysis.csv
