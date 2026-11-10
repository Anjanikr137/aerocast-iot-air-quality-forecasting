"""
Core Training and Model Persistence Pipeline for AeroCast.
Trains and evaluates:
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Artificial Neural Network (Keras Feed-Forward)

Saves models, scalers, imputers, evaluation metrics, and metadata under:
models/<model_name>/
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = os.path.abspath('.cache/keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.evaluation.metrics import evaluate_regression, evaluate_aqi_classification
from src.evaluation.visualization import (
    plot_actual_vs_predicted,
    plot_time_series_comparison,
    plot_residuals,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_linear_coefficients,
    plot_learning_curve
)

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']

def load_processed_data(processed_dir='data/processed'):
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'), index_col=0)
    val_df = pd.read_csv(os.path.join(processed_dir, 'val.csv'), index_col=0)
    test_df = pd.read_csv(os.path.join(processed_dir, 'test.csv'), index_col=0)
    
    with open(os.path.join(processed_dir, 'feature_metadata.json')) as f:
        meta = json.load(f)
        
    feature_cols = meta['feature_names']
    
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    
    # AQI targets
    y_aqi_train = train_df['Target_AQI'].values
    y_aqi_val = val_df['Target_AQI'].values
    y_aqi_test = test_df['Target_AQI'].values
    
    # Pollutant targets
    pollutant_cols = [f'Target_{p}' for p in TARGET_POLLUTANTS]
    y_pol_train = train_df[pollutant_cols].values
    y_pol_val = val_df[pollutant_cols].values
    y_pol_test = test_df[pollutant_cols].values
    
    # Fit imputer and scaler ONLY on training data
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    preprocessing = {
        'imputer': imputer,
        'scaler': scaler,
        'feature_names': feature_cols,
        'target_pollutants': TARGET_POLLUTANTS
    }
    
    datasets = {
        'X_train_raw': X_train_imp,
        'X_val_raw': X_val_imp,
        'X_test_raw': X_test_imp,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_aqi_train': y_aqi_train,
        'y_aqi_val': y_aqi_val,
        'y_aqi_test': y_aqi_test,
        'y_pol_train': y_pol_train,
        'y_pol_val': y_pol_val,
        'y_pol_test': y_pol_test,
        'feature_cols': feature_cols,
        'pollutant_cols': pollutant_cols,
        'test_index': test_df.index
    }
    
    return datasets, preprocessing

# ==========================================
# 1. Linear Regression
# ==========================================
def train_linear_regression(datasets, preprocessing):
    print("\n--- Training Linear Regression ---")
    model_dir = "models/linear_regression"
    os.makedirs(model_dir, exist_ok=True)
    
    X_train = datasets['X_train_scaled']
    X_val = datasets['X_val_scaled']
    X_test = datasets['X_test_scaled']
    
    # A. AQI Model
    aqi_model = LinearRegression()
    aqi_model.fit(X_train, datasets['y_aqi_train'])
    
    # B. Pollutant Model
    pol_model = LinearRegression()
    pol_model.fit(X_train, datasets['y_pol_train'])
    
    # Evaluate AQI
    preds_aqi_train = aqi_model.predict(X_train)
    preds_aqi_val = aqi_model.predict(X_val)
    preds_aqi_test = aqi_model.predict(X_test)
    
    # Evaluate Pollutants
    preds_pol_train = pol_model.predict(X_train)
    preds_pol_val = pol_model.predict(X_val)
    preds_pol_test = pol_model.predict(X_test)
    
    # Metrics
    aqi_reg_train = evaluate_regression(datasets['y_aqi_train'], preds_aqi_train, "AQI")
    aqi_reg_val = evaluate_regression(datasets['y_aqi_val'], preds_aqi_val, "AQI")
    aqi_reg_test = evaluate_regression(datasets['y_aqi_test'], preds_aqi_test, "AQI")
    
    aqi_clf_train = evaluate_aqi_classification(datasets['y_aqi_train'], preds_aqi_train)
    aqi_clf_val = evaluate_aqi_classification(datasets['y_aqi_val'], preds_aqi_val)
    aqi_clf_test = evaluate_aqi_classification(datasets['y_aqi_test'], preds_aqi_test)
    
    pol_metrics_test = {}
    for idx, p in enumerate(TARGET_POLLUTANTS):
        pol_metrics_test[p] = evaluate_regression(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p)
        
    print(f"Linear Regression AQI Test - MAE: {aqi_reg_test['MAE']}, RMSE: {aqi_reg_test['RMSE']}, R2: {aqi_reg_test['R2']}, Acc: {aqi_clf_test['Accuracy']}")
    
    # Visualizations
    plot_actual_vs_predicted(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Linear Regression")
    plot_time_series_comparison(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Linear Regression", datasets['test_index'])
    plot_residuals(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Linear Regression")
    plot_confusion_matrix(aqi_clf_test['Confusion_Matrix'], "Linear Regression")
    plot_linear_coefficients(datasets['feature_cols'], aqi_model.coef_, "Linear Regression")
    
    # Pollutant scatter plots
    for idx, p in enumerate(TARGET_POLLUTANTS):
        plot_actual_vs_predicted(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p, "Linear Regression")
        
    # Save artifacts
    joblib.dump(aqi_model, os.path.join(model_dir, "aqi_model.joblib"))
    joblib.dump(pol_model, os.path.join(model_dir, "pollutant_model.joblib"))
    joblib.dump(preprocessing, os.path.join(model_dir, "preprocessing.joblib"))
    
    metadata = {
        "model_family": "Linear Regression",
        "library": "scikit-learn",
        "version": LinearRegression().__class__.__module__,
        "features": datasets['feature_cols'],
        "aqi_metrics": {
            "train": {**aqi_reg_train, "Accuracy": aqi_clf_train["Accuracy"], "F1_Macro": aqi_clf_train["F1_Macro"]},
            "val": {**aqi_reg_val, "Accuracy": aqi_clf_val["Accuracy"], "F1_Macro": aqi_clf_val["F1_Macro"]},
            "test": {**aqi_reg_test, "Accuracy": aqi_clf_test["Accuracy"], "F1_Macro": aqi_clf_test["F1_Macro"]}
        },
        "pollutant_test_metrics": pol_metrics_test
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata, (preds_aqi_test, preds_pol_test)

# ==========================================
# 2. Random Forest
# ==========================================
def train_random_forest(datasets, preprocessing):
    print("\n--- Training Random Forest ---")
    model_dir = "models/random_forest"
    os.makedirs(model_dir, exist_ok=True)
    
    X_train = datasets['X_train_raw']
    X_val = datasets['X_val_raw']
    X_test = datasets['X_test_raw']
    
    # AQI Model with reasonable hyperparameters
    aqi_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    aqi_model.fit(X_train, datasets['y_aqi_train'])
    
    # Multi-output Pollutant Model
    pol_model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    pol_model.fit(X_train, datasets['y_pol_train'])
    
    # Predictions
    preds_aqi_train = aqi_model.predict(X_train)
    preds_aqi_val = aqi_model.predict(X_val)
    preds_aqi_test = aqi_model.predict(X_test)
    
    preds_pol_train = pol_model.predict(X_train)
    preds_pol_val = pol_model.predict(X_val)
    preds_pol_test = pol_model.predict(X_test)
    
    # Metrics
    aqi_reg_train = evaluate_regression(datasets['y_aqi_train'], preds_aqi_train, "AQI")
    aqi_reg_val = evaluate_regression(datasets['y_aqi_val'], preds_aqi_val, "AQI")
    aqi_reg_test = evaluate_regression(datasets['y_aqi_test'], preds_aqi_test, "AQI")
    
    aqi_clf_train = evaluate_aqi_classification(datasets['y_aqi_train'], preds_aqi_train)
    aqi_clf_val = evaluate_aqi_classification(datasets['y_aqi_val'], preds_aqi_val)
    aqi_clf_test = evaluate_aqi_classification(datasets['y_aqi_test'], preds_aqi_test)
    
    pol_metrics_test = {}
    for idx, p in enumerate(TARGET_POLLUTANTS):
        pol_metrics_test[p] = evaluate_regression(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p)
        
    print(f"Random Forest AQI Test - MAE: {aqi_reg_test['MAE']}, RMSE: {aqi_reg_test['RMSE']}, R2: {aqi_reg_test['R2']}, Acc: {aqi_clf_test['Accuracy']}")
    
    # Visualizations
    plot_actual_vs_predicted(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Random Forest")
    plot_time_series_comparison(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Random Forest", datasets['test_index'])
    plot_residuals(datasets['y_aqi_test'], preds_aqi_test, "AQI", "Random Forest")
    plot_confusion_matrix(aqi_clf_test['Confusion_Matrix'], "Random Forest")
    plot_feature_importance(datasets['feature_cols'], aqi_model.feature_importances_, "Random Forest")
    
    for idx, p in enumerate(TARGET_POLLUTANTS):
        plot_actual_vs_predicted(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p, "Random Forest")
        
    # Save artifacts
    joblib.dump(aqi_model, os.path.join(model_dir, "aqi_model.joblib"))
    joblib.dump(pol_model, os.path.join(model_dir, "pollutant_model.joblib"))
    joblib.dump(preprocessing, os.path.join(model_dir, "preprocessing.joblib"))
    
    metadata = {
        "model_family": "Random Forest",
        "hyperparameters": {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
        "features": datasets['feature_cols'],
        "aqi_metrics": {
            "train": {**aqi_reg_train, "Accuracy": aqi_clf_train["Accuracy"], "F1_Macro": aqi_clf_train["F1_Macro"]},
            "val": {**aqi_reg_val, "Accuracy": aqi_clf_val["Accuracy"], "F1_Macro": aqi_clf_val["F1_Macro"]},
            "test": {**aqi_reg_test, "Accuracy": aqi_clf_test["Accuracy"], "F1_Macro": aqi_clf_test["F1_Macro"]}
        },
        "pollutant_test_metrics": pol_metrics_test
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata, (preds_aqi_test, preds_pol_test)

# ==========================================
# 3. XGBoost
# ==========================================
def train_xgboost(datasets, preprocessing):
    print("\n--- Training XGBoost ---")
    model_dir = "models/xgboost"
    os.makedirs(model_dir, exist_ok=True)
    
    X_train = datasets['X_train_raw']
    X_val = datasets['X_val_raw']
    X_test = datasets['X_test_raw']
    
    # AQI Model with evaluation set for learning curve
    aqi_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    aqi_model.fit(
        X_train, datasets['y_aqi_train'],
        eval_set=[(X_train, datasets['y_aqi_train']), (X_val, datasets['y_aqi_val'])],
        verbose=False
    )
    
    # Pollutant Models (MultiOutput wrapper over XGBRegressor)
    pol_base = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    pol_model = MultiOutputRegressor(pol_base)
    pol_model.fit(X_train, datasets['y_pol_train'])
    
    # Predictions
    preds_aqi_train = aqi_model.predict(X_train)
    preds_aqi_val = aqi_model.predict(X_val)
    preds_aqi_test = aqi_model.predict(X_test)
    
    preds_pol_train = pol_model.predict(X_train)
    preds_pol_val = pol_model.predict(X_val)
    preds_pol_test = pol_model.predict(X_test)
    
    # Metrics
    aqi_reg_train = evaluate_regression(datasets['y_aqi_train'], preds_aqi_train, "AQI")
    aqi_reg_val = evaluate_regression(datasets['y_aqi_val'], preds_aqi_val, "AQI")
    aqi_reg_test = evaluate_regression(datasets['y_aqi_test'], preds_aqi_test, "AQI")
    
    aqi_clf_train = evaluate_aqi_classification(datasets['y_aqi_train'], preds_aqi_train)
    aqi_clf_val = evaluate_aqi_classification(datasets['y_aqi_val'], preds_aqi_val)
    aqi_clf_test = evaluate_aqi_classification(datasets['y_aqi_test'], preds_aqi_test)
    
    pol_metrics_test = {}
    for idx, p in enumerate(TARGET_POLLUTANTS):
        pol_metrics_test[p] = evaluate_regression(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p)
        
    print(f"XGBoost AQI Test - MAE: {aqi_reg_test['MAE']}, RMSE: {aqi_reg_test['RMSE']}, R2: {aqi_reg_test['R2']}, Acc: {aqi_clf_test['Accuracy']}")
    
    # Visualizations
    plot_actual_vs_predicted(datasets['y_aqi_test'], preds_aqi_test, "AQI", "XGBoost")
    plot_time_series_comparison(datasets['y_aqi_test'], preds_aqi_test, "AQI", "XGBoost", datasets['test_index'])
    plot_residuals(datasets['y_aqi_test'], preds_aqi_test, "AQI", "XGBoost")
    plot_confusion_matrix(aqi_clf_test['Confusion_Matrix'], "XGBoost")
    plot_feature_importance(datasets['feature_cols'], aqi_model.feature_importances_, "XGBoost")
    
    # Learning curve
    evals_result = aqi_model.evals_result()
    train_rmse = evals_result['validation_0']['rmse']
    val_rmse = evals_result['validation_1']['rmse']
    plot_learning_curve(train_rmse, val_rmse, "XGBoost", "RMSE")
    
    for idx, p in enumerate(TARGET_POLLUTANTS):
        plot_actual_vs_predicted(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p, "XGBoost")
        
    # Save artifacts
    aqi_model.save_model(os.path.join(model_dir, "aqi_model.json"))
    joblib.dump(aqi_model, os.path.join(model_dir, "aqi_model.joblib"))
    joblib.dump(pol_model, os.path.join(model_dir, "pollutant_model.joblib"))
    joblib.dump(preprocessing, os.path.join(model_dir, "preprocessing.joblib"))
    
    metadata = {
        "model_family": "XGBoost",
        "hyperparameters": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8},
        "features": datasets['feature_cols'],
        "aqi_metrics": {
            "train": {**aqi_reg_train, "Accuracy": aqi_clf_train["Accuracy"], "F1_Macro": aqi_clf_train["F1_Macro"]},
            "val": {**aqi_reg_val, "Accuracy": aqi_clf_val["Accuracy"], "F1_Macro": aqi_clf_val["F1_Macro"]},
            "test": {**aqi_reg_test, "Accuracy": aqi_clf_test["Accuracy"], "F1_Macro": aqi_clf_test["F1_Macro"]}
        },
        "pollutant_test_metrics": pol_metrics_test
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata, (preds_aqi_test, preds_pol_test)

# ==========================================
# 4. Artificial Neural Network (Keras)
# ==========================================
def train_ann(datasets, preprocessing):
    print("\n--- Training Artificial Neural Network (ANN) ---")
    model_dir = "models/ann"
    os.makedirs(model_dir, exist_ok=True)
    
    X_train = datasets['X_train_scaled']
    X_val = datasets['X_val_scaled']
    X_test = datasets['X_test_scaled']
    n_features = X_train.shape[1]
    
    # A. AQI Feed-Forward Network
    def build_ann(output_dim=1):
        model = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    aqi_model = build_ann(output_dim=1)
    
    es_aqi = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    history_aqi = aqi_model.fit(
        X_train, datasets['y_aqi_train'],
        validation_data=(X_val, datasets['y_aqi_val']),
        epochs=80,
        batch_size=64,
        callbacks=[es_aqi],
        verbose=0
    )
    
    # B. Pollutant Multi-output Network
    pol_model = build_ann(output_dim=len(TARGET_POLLUTANTS))
    es_pol = callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    history_pol = pol_model.fit(
        X_train, datasets['y_pol_train'],
        validation_data=(X_val, datasets['y_pol_val']),
        epochs=80,
        batch_size=64,
        callbacks=[es_pol],
        verbose=0
    )
    
    # Predictions
    preds_aqi_train = aqi_model.predict(X_train, verbose=0).flatten()
    preds_aqi_val = aqi_model.predict(X_val, verbose=0).flatten()
    preds_aqi_test = aqi_model.predict(X_test, verbose=0).flatten()
    
    preds_pol_train = pol_model.predict(X_train, verbose=0)
    preds_pol_val = pol_model.predict(X_val, verbose=0)
    preds_pol_test = pol_model.predict(X_test, verbose=0)
    
    # Metrics
    aqi_reg_train = evaluate_regression(datasets['y_aqi_train'], preds_aqi_train, "AQI")
    aqi_reg_val = evaluate_regression(datasets['y_aqi_val'], preds_aqi_val, "AQI")
    aqi_reg_test = evaluate_regression(datasets['y_aqi_test'], preds_aqi_test, "AQI")
    
    aqi_clf_train = evaluate_aqi_classification(datasets['y_aqi_train'], preds_aqi_train)
    aqi_clf_val = evaluate_aqi_classification(datasets['y_aqi_val'], preds_aqi_val)
    aqi_clf_test = evaluate_aqi_classification(datasets['y_aqi_test'], preds_aqi_test)
    
    pol_metrics_test = {}
    for idx, p in enumerate(TARGET_POLLUTANTS):
        pol_metrics_test[p] = evaluate_regression(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p)
        
    print(f"ANN AQI Test - MAE: {aqi_reg_test['MAE']}, RMSE: {aqi_reg_test['RMSE']}, R2: {aqi_reg_test['R2']}, Acc: {aqi_clf_test['Accuracy']}")
    
    # Visualizations
    plot_actual_vs_predicted(datasets['y_aqi_test'], preds_aqi_test, "AQI", "ANN")
    plot_time_series_comparison(datasets['y_aqi_test'], preds_aqi_test, "AQI", "ANN", datasets['test_index'])
    plot_residuals(datasets['y_aqi_test'], preds_aqi_test, "AQI", "ANN")
    plot_confusion_matrix(aqi_clf_test['Confusion_Matrix'], "ANN")
    
    # Learning curves
    plot_learning_curve(history_aqi.history['loss'], history_aqi.history['val_loss'], "ANN", "Loss (MSE)")
    plot_learning_curve(history_aqi.history['mae'], history_aqi.history['val_mae'], "ANN", "MAE")
    
    for idx, p in enumerate(TARGET_POLLUTANTS):
        plot_actual_vs_predicted(datasets['y_pol_test'][:, idx], preds_pol_test[:, idx], p, "ANN")
        
    # Save artifacts
    aqi_model.save(os.path.join(model_dir, "aqi_model.keras"))
    pol_model.save(os.path.join(model_dir, "pollutant_model.keras"))
    joblib.dump(preprocessing, os.path.join(model_dir, "preprocessing.joblib"))
    
    metadata = {
        "model_family": "Artificial Neural Network",
        "framework": "TensorFlow/Keras",
        "architecture": "Feed-Forward [128 -> 64 -> 32 -> Output]",
        "optimizer": "Adam (lr=0.001)",
        "features": datasets['feature_cols'],
        "aqi_metrics": {
            "train": {**aqi_reg_train, "Accuracy": aqi_clf_train["Accuracy"], "F1_Macro": aqi_clf_train["F1_Macro"]},
            "val": {**aqi_reg_val, "Accuracy": aqi_clf_val["Accuracy"], "F1_Macro": aqi_clf_val["F1_Macro"]},
            "test": {**aqi_reg_test, "Accuracy": aqi_clf_test["Accuracy"], "F1_Macro": aqi_clf_test["F1_Macro"]}
        },
        "pollutant_test_metrics": pol_metrics_test
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata, (preds_aqi_test, preds_pol_test)

