# Wafer Project Comparison

## Reviewed References

| Reference | Useful pattern | Applied in this project |
| --- | --- | --- |
| MathWorks wafer map deep learning example: https://www.mathworks.com/help/vision/ug/classify-defects-on-wafer-maps-using-deep-learning.html | WM-811K uses pixel values 0/1/2, heavily imbalanced labels, preprocessing, augmentation, and test-set evaluation. | Kept the same pixel convention, added train/validation/test tracking, class-weighted loss, augmentation, and per-class reports. |
| DMkelllog MultiNN: https://github.com/DMkelllog/wafermap_MultiNN | Combines CNN features with handcrafted/domain features and uses 64x64 resized wafer maps. | Added the artifact structure needed for deeper model comparisons. Full handcrafted-feature fusion remains a future model option. |
| DMkelllog handcrafted features: https://github.com/DMkelllog/wafermap_handcrafted_features | Highlights classical wafer-map feature engineering and the WM-811K class taxonomy. | Added per-class evaluation so class-specific weaknesses are visible before adding handcrafted features. |
| Stanford CS230 wafer map report: https://cs230.stanford.edu/projects_fall_2019/reports/26259703.pdf | Uses normalization, resizing, augmentation, validation tuning, and compares CNN variants. | Added validation-based best checkpoint selection, augmentation, training history, and best validation F1. |
| Autoencoder augmentation paper: https://arxiv.org/html/2411.11029v1 | Focuses on augmentation, imbalance mitigation, and interpretability for class-specific failures. | Added class-weighted learning and class-level UI/reporting. Autoencoder augmentation and explainability are left as future upgrades. |

## Implemented Improvements

| Area | Before | After |
| --- | --- | --- |
| Data input | Synthetic/pickle-centric loader | `.pkl`, `.csv`, `.json`, `.jsonl`, JSON wafer maps, and `.npy` map paths |
| Training split | Train/test only | Train/validation/test, with original train split preserved |
| Imbalance handling | Plain cross-entropy | Class-weighted cross-entropy |
| Augmentation | None | Training-only random rotations and flips |
| Optimization | Adam, final checkpoint | AdamW, validation F1 scheduler, best checkpoint |
| Monitoring artifacts | Metrics JSON and confusion matrix | Metrics JSON, confusion matrix, class report, training history, model registry |
| Dashboard | Aggregate metrics and confusion matrix | Adds training history and per-class F1/recall chart/table |
| API operations | Single prediction and summary APIs | Adds batch job tracking, model registry, and Prometheus-style metrics |

## Current Demo Result

Synthetic demo test-set result after the updated pipeline:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.9028 |
| Weighted F1 | 0.8908 |
| Best validation F1 | 0.8775 |

The lowest class F1 in the latest run is `Loc`, so future model work should focus on localized-cluster defect augmentation or handcrafted spatial-density features.
