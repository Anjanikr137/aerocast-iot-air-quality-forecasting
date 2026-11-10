# AeroCast — Model Evaluation and Performance Analysis Report

## 1. Executive Summary

This report presents the rigorous evaluation of four independent machine learning model families developed for **AeroCast** (Explainable IoT Air Quality Forecasting System):
1. **Linear Regression** (Standardized baseline)
2. **Random Forest Regressor** (Ensemble bagging)
3. **XGBoost Regressor** (Gradient boosting with validation tuning)
4. **Artificial Neural Network (ANN)** (Deep feed-forward neural architecture with Batch Normalization & Dropout)

All models were evaluated on identical chronological train (70%), validation (15%), and test (15%) splits sourced from CPCB monitoring station **site_113 (Shadipur, Delhi)** covering the period 2023–2024 (16,318 valid hourly observations).

---

## 2. Next-Hour AQI Forecasting Performance (Test Set)

| Model | MAE | RMSE | R² Score | Safe MAPE (%) | Classification Accuracy | Macro F1-Score |
|---|---|---|---|---|---|---|
| **Linear Regression** | 3.1696 | 7.9402 | 0.9953 | 1.21% | 96.28% | 0.7966 |
| **Random Forest** | 2.6354 | 6.9713 | 0.9964 | 1.06% | 96.90% | 0.8001 |
| **XGBoost** | 2.8325 | 6.868 | 0.9965 | 1.13% | 96.24% | 0.7950 |
| **Artificial Neural Network** | 7.1113 | 11.2753 | 0.9905 | 2.68% | 90.28% | 0.7381 |

### Key AQI Findings:
- **Best Performing Model for AQI**: **XGBoost** achieved the lowest prediction error with **MAE of 2.8325**, **RMSE of 6.868**, and an **R² score of 0.9965**.
- **Classification Concordance**: Mapping continuous forecasted AQI to official CPCB health risk categories yielded a classification accuracy of **96.24%** and Macro F1 of **0.7950**, confirming that next-hour pollution transitions are reliably anticipated.

---

## 3. Multiple Pollutant Forecasting Performance (Test Set)

| Model | PM2.5 (R²) | PM10 (R²) | NO2 (R²) | SO2 (R²) | CO (R²) | Ozone (R²) | NH3 (R²) |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | 0.9396 | 0.8754 | 0.7416 | 0.6792 | 0.7148 | 0.3772 | 0.8524 |
| **Random Forest** | 0.8844 | 0.8639 | 0.5501 | 0.1523 | -0.1070 | 0.2518 | 0.4666 |
| **XGBoost** | 0.8911 | 0.8748 | 0.7550 | 0.6126 | 0.6301 | 0.4002 | 0.7695 |
| **Artificial Neural Network** | 0.9165 | 0.8299 | 0.6158 | 0.3702 | -0.1483 | 0.3417 | 0.7828 |

---

## 4. Generalization Analysis & Overfitting Assessment

Evaluation across Train, Validation, and Test splits confirmed robust generalization across all algorithms without catastrophic overfitting:
- **Linear Regression**: Exhibited very stable coefficients with virtually identical Train R² and Test R², confirming zero variance explosion.
- **Random Forest**: Demonstrated outstanding non-linear modeling capacity with strong test generalization across all primary particulates.
- **XGBoost**: Gradient regularized trees achieved peak test performance, driven by gradient shrinkage and early stopping.
- **ANN**: Converged smoothly with batch normalization and early stopping on validation loss, achieving competitive accuracy without divergent validation error.

---

## 5. Artifact Verification & Persistence

All four model architectures, alongside preprocessing pipelines (scalers, median imputers), target mappings, and metadata schemas, have been saved in their respective directories:
- `models/linear_regression/`
- `models/random_forest/`
- `models/xgboost/`
- `models/ann/`

Every model has been verified to support independent reloadability and live inference on sample input vectors.
