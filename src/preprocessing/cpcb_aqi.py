"""
Official CPCB National Air Quality Index (NAQI) Calculation Module.
Implements the Indian Central Pollution Control Board (CPCB) methodology:
- 24-hour averaging for PM2.5, PM10, NO2, SO2, NH3
- 8-hour averaging for CO and Ozone
- Linear interpolation for sub-index calculation
- Eligibility check: at least 3 pollutants present, with at least one PM2.5 or PM10
- Overall AQI = maximum of eligible sub-indices
- Categorization into 6 standard CPCB bands:
  Good (0-50), Satisfactory (51-100), Moderate (101-200),
  Poor (201-300), Very Poor (301-400), Severe (401-500)
"""

import numpy as np
import pandas as pd

# CPCB Breakpoints: (B_lo, B_hi, I_lo, I_hi)
BREAKPOINTS = {
    'PM2.5': [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500)
    ],
    'PM10': [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500)
    ],
    'NO2': [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 600.0, 401, 500)
    ],
    'SO2': [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2000.0, 401, 500)
    ],
    'CO': [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500)
    ],
    'Ozone': [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500)
    ],
    'NH3': [
        (0.0, 200.0, 0, 50),
        (200.1, 400.0, 51, 100),
        (401.0, 800.0, 101, 200),
        (801.0, 1200.0, 201, 300),
        (1201.0, 1800.0, 301, 400),
        (1801.0, 2400.0, 401, 500)
    ]
}

AQI_CATEGORIES = [
    (0, 50, 'Good'),
    (51, 100, 'Satisfactory'),
    (101, 200, 'Moderate'),
    (201, 300, 'Poor'),
    (301, 400, 'Very Poor'),
    (401, float('inf'), 'Severe')
]

CATEGORY_NAMES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

def calculate_sub_index(conc, pollutant):
    """
    Calculate sub-index for a given pollutant concentration using linear interpolation.
    """
    if pd.isna(conc) or conc < 0:
        return np.nan
    
    table = BREAKPOINTS.get(pollutant)
    if not table:
        return np.nan
    
    for b_lo, b_hi, i_lo, i_hi in table:
        if b_lo <= conc <= b_hi:
            return i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (conc - b_lo)
            
    # If concentration exceeds highest breakpoint, extrapolate based on highest bracket
    b_lo, b_hi, i_lo, i_hi = table[-1]
    if conc > b_hi:
        sub = i_lo + ((i_hi - i_lo) / (b_hi - b_lo)) * (conc - b_lo)
        return min(sub, 500.0)
        
    return np.nan

def aqi_to_category(aqi_val):
    """
    Convert numerical AQI into official CPCB Category label.
    """
    if pd.isna(aqi_val) or aqi_val < 0:
        return np.nan
    for low, high, cat in AQI_CATEGORIES:
        if low <= aqi_val <= high:
            return cat
    return 'Severe'

def compute_cpcb_aqi_dataframe(df_hourly, min_coverage_ratio=0.75):
    """
    Given a chronological hourly DataFrame with columns:
    ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3'] indexed by timestamp:
    1. Computes 24h rolling mean for PM2.5, PM10, NO2, SO2, NH3 (min_periods = int(24 * min_coverage_ratio) = 18 hrs).
    2. Computes 8h rolling mean for CO and Ozone (min_periods = int(8 * min_coverage_ratio) = 6 hrs).
    3. Calculates sub-index for each pollutant.
    4. Enforces eligibility: at least 3 sub-indices present, with at least one PM2.5 or PM10.
    5. Overall AQI = max(sub-indices).
    """
    df = df_hourly.copy()
    
    # Required rolling windows
    pollutants_24h = [p for p in ['PM2.5', 'PM10', 'NO2', 'SO2', 'NH3'] if p in df.columns]
    pollutants_8h = [p for p in ['CO', 'Ozone'] if p in df.columns]
    
    sub_df = pd.DataFrame(index=df.index)
    
    for p in pollutants_24h:
        # 24h backward rolling average (past 24 hours including current)
        rolling_avg = df[p].rolling(window=24, min_periods=int(24 * min_coverage_ratio)).mean()
        sub_df[f'{p}_sub'] = rolling_avg.apply(lambda x: calculate_sub_index(x, p))
        
    for p in pollutants_8h:
        # 8h backward rolling average
        rolling_avg = df[p].rolling(window=8, min_periods=int(8 * min_coverage_ratio)).mean()
        sub_df[f'{p}_sub'] = rolling_avg.apply(lambda x: calculate_sub_index(x, p))
        
    # Check eligibility row by row
    sub_cols = [c for c in sub_df.columns if c.endswith('_sub')]
    
    def evaluate_aqi_row(row):
        valid_subs = row.dropna()
        if len(valid_subs) < 3:
            return np.nan
        has_pm = ('PM2.5_sub' in valid_subs.index) or ('PM10_sub' in valid_subs.index)
        if not has_pm:
            return np.nan
        return valid_subs.max()
        
    aqi_series = sub_df.apply(evaluate_aqi_row, axis=1)
    sub_df['AQI'] = aqi_series
    sub_df['AQI_Category'] = aqi_series.apply(aqi_to_category)
    
    return sub_df

