"""
Comprehensive Model Verification Script for AeroCast.
Verifies independent reloadability and sample prediction across all eight saved models:
1. Linear Regression - AQI Model
2. Linear Regression - Multi-Pollutant Model
3. Random Forest - AQI Model
4. Random Forest - Multi-Pollutant Model
5. XGBoost - AQI Model
6. XGBoost - Multi-Pollutant Model
7. Artificial Neural Network - AQI Model
8. Artificial Neural Network - Multi-Pollutant Model
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

os.environ['KERAS_HOME'] = os.path.abspath('.cache/keras')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']
POLLUTANT_UNITS = {
    'PM2.5': 'µg/m³',
    'PM10': 'µg/m³',
    'NO2': 'µg/m³',
    'SO2': 'µg/m³',
    'CO': 'mg/m³',
    'Ozone': 'µg/m³',
    'NH3': 'µg/m³'
}

def verify_all_models():
    print("=" * 70)
    print("AEROCAST: INDEPENDENT VERIFICATION OF ALL 8 SAVED PREDICTION MODELS")
    print("=" * 70)
    
    # Load test vector
    test_df = pd.read_csv("data/processed/test.csv", index_col=0)
    with open("data/processed/feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta['feature_names']
    
    sample_raw = test_df[feature_cols].iloc[0:1]
    actual_aqi = test_df['Target_AQI'].iloc[0]
    actual_pollutants = {p: test_df[f'Target_{p}'].iloc[0] for p in TARGET_POLLUTANTS}
    
    print(f"\nEvaluation Benchmark (Actual Next-Hour Values for Test Sample 0):")
    print(f"  Timestamp: {test_df.index[0]}")
    print(f"  Actual Next-Hour AQI: {actual_aqi:.2f}")
    pol_str = ", ".join([f"{p}: {actual_pollutants[p]:.2f} {POLLUTANT_UNITS[p]}" for p in TARGET_POLLUTANTS])
    print(f"  Actual Next-Hour Pollutants: {pol_str}\n")
    
    results = []

    # -------------------------------------------------------------
    # 1 & 2. Linear Regression
    # -------------------------------------------------------------
    print("[1/4] Verifying Linear Regression Models...")
    lr_dir = "models/linear_regression"
    lr_prep = joblib.load(os.path.join(lr_dir, "preprocessing.joblib"))
    lr_aqi_model = joblib.load(os.path.join(lr_dir, "aqi_model.joblib"))
    lr_pol_model = joblib.load(os.path.join(lr_dir, "pollutant_model.joblib"))
    
    sample_lr = lr_prep['scaler'].transform(lr_prep['imputer'].transform(sample_raw))
    pred_lr_aqi = float(lr_aqi_model.predict(sample_lr)[0])
    pred_lr_pol = lr_pol_model.predict(sample_lr)[0]
    
    print(f"  -> [OK] Model 1 (Linear Regression AQI): Predicted AQI = {pred_lr_aqi:.2f}")
    print(f"  -> [OK] Model 2 (Linear Regression Multi-Pollutant): Predicted PM2.5 = {pred_lr_pol[0]:.2f} µg/m³, PM10 = {pred_lr_pol[1]:.2f} µg/m³")
    results.append({
        "Model Family": "Linear Regression",
        "AQI Model File": "models/linear_regression/aqi_model.joblib",
        "Pollutant Model File": "models/linear_regression/pollutant_model.joblib",
        "Predicted Sample AQI": round(pred_lr_aqi, 2),
        "Predicted Sample PM2.5": round(float(pred_lr_pol[0]), 2),
        "Verification Status": "VERIFIED & OPERATIONAL"
    })

    # -------------------------------------------------------------
    # 3 & 4. Random Forest
    # -------------------------------------------------------------
    print("\n[2/4] Verifying Random Forest Models...")
    rf_dir = "models/random_forest"
    rf_prep = joblib.load(os.path.join(rf_dir, "preprocessing.joblib"))
    rf_aqi_model = joblib.load(os.path.join(rf_dir, "aqi_model.joblib"))
    rf_pol_model = joblib.load(os.path.join(rf_dir, "pollutant_model.joblib"))
    
    sample_rf = rf_prep['imputer'].transform(sample_raw)
    pred_rf_aqi = float(rf_aqi_model.predict(sample_rf)[0])
    pred_rf_pol = rf_pol_model.predict(sample_rf)[0]
    
    print(f"  -> [OK] Model 3 (Random Forest AQI): Predicted AQI = {pred_rf_aqi:.2f}")
    print(f"  -> [OK] Model 4 (Random Forest Multi-Pollutant): Predicted PM2.5 = {pred_rf_pol[0]:.2f} µg/m³, PM10 = {pred_rf_pol[1]:.2f} µg/m³")
    results.append({
        "Model Family": "Random Forest",
        "AQI Model File": "models/random_forest/aqi_model.joblib",
        "Pollutant Model File": "models/random_forest/pollutant_model.joblib",
        "Predicted Sample AQI": round(pred_rf_aqi, 2),
        "Predicted Sample PM2.5": round(float(pred_rf_pol[0]), 2),
        "Verification Status": "VERIFIED & OPERATIONAL"
    })

    # -------------------------------------------------------------
    # 5 & 6. XGBoost
    # -------------------------------------------------------------
    print("\n[3/4] Verifying XGBoost Models...")
    xgb_dir = "models/xgboost"
    xgb_prep = joblib.load(os.path.join(xgb_dir, "preprocessing.joblib"))
    xgb_aqi_model = joblib.load(os.path.join(xgb_dir, "aqi_model.joblib"))
    xgb_pol_model = joblib.load(os.path.join(xgb_dir, "pollutant_model.joblib"))
    
    sample_xgb = xgb_prep['imputer'].transform(sample_raw)
    pred_xgb_aqi = float(xgb_aqi_model.predict(sample_xgb)[0])
    pred_xgb_pol = xgb_pol_model.predict(sample_xgb)[0]
    
    print(f"  -> [OK] Model 5 (XGBoost AQI): Predicted AQI = {pred_xgb_aqi:.2f}")
    print(f"  -> [OK] Model 6 (XGBoost Multi-Pollutant): Predicted PM2.5 = {pred_xgb_pol[0]:.2f} µg/m³, PM10 = {pred_xgb_pol[1]:.2f} µg/m³")
    results.append({
        "Model Family": "XGBoost",
        "AQI Model File": "models/xgboost/aqi_model.joblib (and .json)",
        "Pollutant Model File": "models/xgboost/pollutant_model.joblib",
        "Predicted Sample AQI": round(pred_xgb_aqi, 2),
        "Predicted Sample PM2.5": round(float(pred_xgb_pol[0]), 2),
        "Verification Status": "VERIFIED & OPERATIONAL"
    })

    # -------------------------------------------------------------
    # 7 & 8. Artificial Neural Network (Keras)
    # -------------------------------------------------------------
    print("\n[4/4] Verifying Artificial Neural Network (ANN) Models...")
    ann_dir = "models/ann"
    ann_prep = joblib.load(os.path.join(ann_dir, "preprocessing.joblib"))
    ann_aqi_model = keras.models.load_model(os.path.join(ann_dir, "aqi_model.keras"))
    ann_pol_model = keras.models.load_model(os.path.join(ann_dir, "pollutant_model.keras"))
    
    sample_ann = ann_prep['scaler'].transform(ann_prep['imputer'].transform(sample_raw))
    pred_ann_aqi = float(ann_aqi_model.predict(sample_ann, verbose=0)[0][0])
    pred_ann_pol = ann_pol_model.predict(sample_ann, verbose=0)[0]
    
    print(f"  -> [OK] Model 7 (ANN AQI): Predicted AQI = {pred_ann_aqi:.2f}")
    print(f"  -> [OK] Model 8 (ANN Multi-Pollutant): Predicted PM2.5 = {pred_ann_pol[0]:.2f} µg/m³, PM10 = {pred_ann_pol[1]:.2f} µg/m³")
    results.append({
        "Model Family": "Artificial Neural Network",
        "AQI Model File": "models/ann/aqi_model.keras",
        "Pollutant Model File": "models/ann/pollutant_model.keras",
        "Predicted Sample AQI": round(pred_ann_aqi, 2),
        "Predicted Sample PM2.5": round(float(pred_ann_pol[0]), 2),
        "Verification Status": "VERIFIED & OPERATIONAL"
    })

    print("\n" + "=" * 70)
    print("FINAL SUMMARY: ALL 8 PREDICTION MODELS SUCCESSFULLY VERIFIED")
    print("=" * 70)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    os.makedirs("reports/model_evaluation", exist_ok=True)
    res_df.to_csv("reports/model_evaluation/eight_models_verification_summary.csv", index=False)
    print("\nSaved verification summary to reports/model_evaluation/eight_models_verification_summary.csv")

if __name__ == '__main__':
    verify_all_models()

