"""
Evaluation Metrics Framework for AeroCast.
Implements:
1. Regression metrics: MAE, RMSE, R², MAPE (with zero-safe handling).
2. AQI Classification metrics: Accuracy, Precision, Recall, F1 (macro, weighted, per-class),
   Confusion Matrix, and Classification Report based on CPCB AQI categories.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.preprocessing.cpcb_aqi import aqi_to_category, CATEGORY_NAMES

def safe_mape(y_true, y_pred, epsilon=1.0):
    """
    Computes MAPE with safe handling for zero and near-zero values.
    Uses denominator max(|y_true|, epsilon).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def evaluate_regression(y_true, y_pred, target_name="AQI"):
    """
    Calculates MAE, RMSE, R², and safe MAPE for continuous predictions.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    
    return {
        "Target": target_name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2)
    }

def evaluate_aqi_classification(y_true_cont, y_pred_cont):
    """
    Maps continuous AQI to CPCB categories and evaluates classification performance.
    """
    # Convert continuous values to discrete categories
    y_true_cat = [aqi_to_category(val) for val in y_true_cont]
    y_pred_cat = [aqi_to_category(val) for val in y_pred_cont]
    
    acc = accuracy_score(y_true_cat, y_pred_cat)
    prec_macro = precision_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="weighted", zero_division=0)
    
    rec_macro = recall_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="weighted", zero_division=0)
    
    f1_macro = f1_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average="weighted", zero_division=0)
    
    # Per-class metrics
    prec_per_class = precision_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average=None, zero_division=0)
    rec_per_class = recall_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES)
    
    per_class_df = pd.DataFrame({
        "Category": CATEGORY_NAMES,
        "Precision": np.round(prec_per_class, 4),
        "Recall": np.round(rec_per_class, 4),
        "F1": np.round(f1_per_class, 4)
    })
    
    clf_report = classification_report(y_true_cat, y_pred_cat, labels=CATEGORY_NAMES, zero_division=0, output_dict=True)
    
    return {
        "Accuracy": round(float(acc), 4),
        "Precision_Macro": round(float(prec_macro), 4),
        "Precision_Weighted": round(float(prec_weighted), 4),
        "Recall_Macro": round(float(rec_macro), 4),
        "Recall_Weighted": round(float(rec_weighted), 4),
        "F1_Macro": round(float(f1_macro), 4),
        "F1_Weighted": round(float(f1_weighted), 4),
        "Confusion_Matrix": cm.tolist(),
        "Per_Class_Metrics": per_class_df,
        "Classification_Report": clf_report,
        "y_true_cat": y_true_cat,
        "y_pred_cat": y_pred_cat
    }

