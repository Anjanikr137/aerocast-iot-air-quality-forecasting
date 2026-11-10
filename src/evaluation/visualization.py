"""
Visualization Framework for AeroCast Model Evaluation.
Generates all required plots:
- Actual vs Predicted scatter plots
- Actual vs Predicted time-series plots
- Residual scatter plots & histograms
- Confusion matrix heatmaps for CPCB categories
- Feature importances and regression coefficients
- Learning curves (XGBoost, ANN)
- Comparative evaluation bar charts
Saves all figures into reports/model_evaluation/plots/
"""

import os
import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.preprocessing.cpcb_aqi import CATEGORY_NAMES

PLOTS_DIR = "reports/model_evaluation/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'figure.autolayout': True, 'savefig.dpi': 300})

def plot_actual_vs_predicted(y_true, y_pred, target_name, model_name, split_name="Test"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.35, edgecolors='none', color='#1f77b4', s=25)
    
    # 45-degree reference line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color='#d62728', linestyle='--', linewidth=2, label='Ideal Fit (y = x)')
    
    ax.set_title(f"{model_name} — Actual vs Predicted {target_name} ({split_name} Set)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(f"Actual {target_name}", fontsize=11)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=11)
    ax.legend(frameon=True)
    
    filename = f"{model_name.lower().replace(' ', '_')}_{target_name.lower()}_{split_name.lower()}_scatter.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_time_series_comparison(y_true, y_pred, target_name, model_name, timestamps=None, n_points=250):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Select last n_points for clear temporal visualization
    y_t = y_true[-n_points:]
    y_p = y_pred[-n_points:]
    x = range(len(y_t)) if timestamps is None else timestamps[-n_points:]
    
    ax.plot(x, y_t, label=f"Actual {target_name}", color='#2ca02c', linewidth=2.0)
    ax.plot(x, y_p, label=f"Predicted {target_name}", color='#ff7f0e', linestyle='--', linewidth=1.8)
    
    ax.set_title(f"{model_name} — Next-Hour {target_name} Forecast vs Actual (Sample Period)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Observation Index / Time", fontsize=11)
    ax.set_ylabel(f"{target_name} Concentration", fontsize=11)
    ax.legend(frameon=True)
    
    filename = f"{model_name.lower().replace(' ', '_')}_{target_name.lower()}_timeseries.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_residuals(y_true, y_pred, target_name, model_name, split_name="Test"):
    residuals = np.array(y_true) - np.array(y_pred)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.35, color='#9467bd', s=25)
    ax1.axhline(0, color='#d62728', linestyle='--', linewidth=1.8)
    ax1.set_title(f"Residuals vs Predicted ({target_name})", fontsize=12, fontweight='bold')
    ax1.set_xlabel(f"Predicted {target_name}", fontsize=11)
    ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
    
    # Residual Distribution
    sns.histplot(residuals, kde=True, ax=ax2, color='#17becf', bins=35)
    ax2.axvline(0, color='#d62728', linestyle='--', linewidth=1.8)
    ax2.set_title(f"Residual Distribution ({target_name})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Residual Value", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    
    fig.suptitle(f"{model_name} Error Analysis — {target_name} ({split_name} Set)", fontsize=14, fontweight='bold', y=1.02)
    
    filename = f"{model_name.lower().replace(' ', '_')}_{target_name.lower()}_residuals.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_confusion_matrix(cm, model_name, split_name="Test"):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CATEGORY_NAMES,
        yticklabels=CATEGORY_NAMES,
        cbar=True,
        ax=ax,
        linewidths=0.5
    )
    ax.set_title(f"{model_name} — AQI CPCB Category Confusion Matrix ({split_name} Set)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted CPCB Category", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual CPCB Category", fontsize=11, labelpad=10)
    
    filename = f"{model_name.lower().replace(' ', '_')}_aqi_confusion_matrix_{split_name.lower()}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_feature_importance(feature_names, importances, model_name, top_n=15):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = [importances[i] for i in indices]
    
    ax.barh(range(top_n, 0, -1), top_importances[::-1], color='#3b528b', align='center')
    ax.set_yticks(range(top_n, 0, -1))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_title(f"{model_name} — Top {top_n} Predictive Features", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Relative Importance", fontsize=11)
    
    filename = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_linear_coefficients(feature_names, coefs, model_name="Linear Regression", top_n=15):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by absolute magnitude
    abs_indices = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_features = [feature_names[i] for i in abs_indices]
    top_coefs = [coefs[i] for i in abs_indices]
    colors = ['#2ca02c' if c > 0 else '#d62728' for c in top_coefs]
    
    ax.barh(range(top_n, 0, -1), top_coefs[::-1], color=colors[::-1], align='center')
    ax.set_yticks(range(top_n, 0, -1))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"{model_name} — Top {top_n} Feature Coefficients (Scaled)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Coefficient Weight", fontsize=11)
    
    filename = "linear_regression_coefficients.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

def plot_learning_curve(train_losses, val_losses, model_name, loss_metric="Loss"):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label=f"Train {loss_metric}", color='#1f77b4', linewidth=2)
    ax.plot(epochs, val_losses, label=f"Validation {loss_metric}", color='#d62728', linestyle='--', linewidth=2)
    
    ax.set_title(f"{model_name} — Learning Curve ({loss_metric} vs Epochs/Iterations)", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Epoch / Iteration", fontsize=11)
    ax.set_ylabel(loss_metric, fontsize=11)
    ax.legend(frameon=True)
    
    filename = f"{model_name.lower().replace(' ', '_')}_learning_curve.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath

