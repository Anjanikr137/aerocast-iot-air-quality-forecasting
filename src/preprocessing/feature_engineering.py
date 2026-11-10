"""
Feature Engineering and Dataset Preparation Pipeline for AeroCast.
Implements:
1. Hourly resampling and chronological indexing.
2. CPCB NAQI calculation with valid averaging windows.
3. Temporal and cyclic features (hour_sin, hour_cos, month_sin, month_cos, day, dayofweek, is_weekend).
4. Historical lag features (1, 2, 3, 6, 12, 24 hours) for all pollutants and AQI.
5. Rolling statistics (3, 6, 12, 24-hour rolling mean and standard deviation) using past observations only.
6. Target creation for next-hour AQI and next-hour pollutant concentrations (t+1).
7. Strict chronological splitting: 70% Train, 15% Validation, 15% Test (No data leakage, fitted on train only).
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath('.'))

import json
import numpy as np
import pandas as pd
from src.preprocessing.cpcb_aqi import compute_cpcb_aqi_dataframe, aqi_to_category, CATEGORY_NAMES

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']
TARGET_COLS = ['Target_AQI'] + [f'Target_{p}' for p in TARGET_POLLUTANTS]

def load_and_pivot_raw(raw_csv_path='data/raw/site_113_raw.csv'):
    df_raw = pd.read_csv(raw_csv_path)
    df_raw['collected_at'] = pd.to_datetime(df_raw['collected_at'])
    pivot = df_raw.pivot_table(index='collected_at', columns='parameter_name', values='value', aggfunc='mean')
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='1h')
    pivot = pivot.reindex(full_idx)
    pivot.index.name = 'timestamp'
    return pivot

def build_features(df_pivot):
    df = df_pivot.copy()
    
    # Calculate CPCB AQI
    aqi_df = compute_cpcb_aqi_dataframe(df)
    df['AQI'] = aqi_df['AQI']
    
    # 1. Clean short isolated missing gaps (up to 3 consecutive hours only)
    for col in TARGET_POLLUTANTS + ['AQI']:
        if col in df.columns:
            df[col] = df[col].interpolate(method='time', limit=3)
            
    # 2. Temporal & Cyclical Features
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12.0)
    
    # 3. Lag Features (1, 2, 3, 6, 12, 24 hours)
    lag_cols = TARGET_POLLUTANTS + ['AQI']
    lags = [1, 2, 3, 6, 12, 24]
    
    for col in lag_cols:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    # 4. Rolling Statistics (Past and current observations only: 3, 6, 12, 24h)
    windows = [3, 6, 12, 24]
    for col in lag_cols:
        for w in windows:
            df[f'{col}_roll_mean_{w}h'] = df[col].rolling(window=w, min_periods=w//2).mean()
        df[f'{col}_roll_std_6h'] = df[col].rolling(window=6, min_periods=3).std()
        df[f'{col}_roll_std_24h'] = df[col].rolling(window=24, min_periods=12).std()
        
    # 5. Create Next-Hour Targets (t+1)
    df['Target_AQI'] = df['AQI'].shift(-1)
    for p in TARGET_POLLUTANTS:
        df[f'Target_{p}'] = df[p].shift(-1)
        
    # Map target AQI to category
    df['Target_AQI_Category'] = df['Target_AQI'].apply(aqi_to_category)
    
    # Drop rows where any of the critical targets are missing
    clean_df = df.dropna(subset=['Target_AQI'] + [f'Target_{p}' for p in TARGET_POLLUTANTS]).copy()
    
    return clean_df

def split_and_save(clean_df, output_dir='data/processed'):
    os.makedirs(output_dir, exist_ok=True)
    
    n_total = len(clean_df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_df = clean_df.iloc[:n_train].copy()
    val_df = clean_df.iloc[n_train:n_train + n_val].copy()
    test_df = clean_df.iloc[n_train + n_val:].copy()
    
    # Identify feature columns (exclude targets, raw timestamp targets, and string category)
    exclude_cols = TARGET_COLS + ['Target_AQI_Category']
    feature_cols = [c for c in clean_df.columns if c not in exclude_cols]
    
    print(f"Total processed rows: {n_total}")
    print(f"Train split: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%) from {train_df.index.min()} to {train_df.index.max()}")
    print(f"Val split:   {len(val_df)} rows ({len(val_df)/n_total*100:.1f}%) from {val_df.index.min()} to {val_df.index.max()}")
    print(f"Test split:  {len(test_df)} rows ({len(test_df)/n_total*100:.1f}%) from {test_df.index.min()} to {test_df.index.max()}")
    print(f"Total features: {len(feature_cols)}")
    
    # Save datasets
    clean_df.to_csv(os.path.join(output_dir, 'full_features.csv'))
    train_df.to_csv(os.path.join(output_dir, 'train.csv'))
    val_df.to_csv(os.path.join(output_dir, 'val.csv'))
    test_df.to_csv(os.path.join(output_dir, 'test.csv'))
    
    # Save metadata
    meta = {
        'total_rows': n_total,
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'test_rows': len(test_df),
        'train_period': [str(train_df.index.min()), str(train_df.index.max())],
        'val_period': [str(val_df.index.min()), str(val_df.index.max())],
        'test_period': [str(test_df.index.min()), str(test_df.index.max())],
        'feature_names': feature_cols,
        'target_names': TARGET_COLS,
        'target_pollutants': TARGET_POLLUTANTS,
        'cpcb_categories': CATEGORY_NAMES
    }
    
    with open(os.path.join(output_dir, 'feature_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print(f"Saved processed data splits and metadata in {output_dir}")
    return train_df, val_df, test_df, feature_cols

def run_pipeline():
    pivot = load_and_pivot_raw()
    clean_df = build_features(pivot)
    return split_and_save(clean_df)

if __name__ == '__main__':
    run_pipeline()
