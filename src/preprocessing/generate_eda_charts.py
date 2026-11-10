"""
Comprehensive Exploratory Data Analysis (EDA) and Chart Generation for AeroCast.
Generates all 10 required EDA figures and summary tables, saving them to reports/eda/
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.abspath('.cache/matplotlib')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.preprocessing.feature_engineering import load_and_pivot_raw
from src.preprocessing.cpcb_aqi import compute_cpcb_aqi_dataframe, aqi_to_category, CATEGORY_NAMES

EDA_DIR = "reports/eda"
os.makedirs(EDA_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'figure.autolayout': True, 'savefig.dpi': 300})

TARGET_POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'NH3']

def generate_eda_figures():
    print("Loading pivoted raw dataset for Shadipur, Delhi (site_113)...")
    pivot = load_and_pivot_raw()
    aqi_df = compute_cpcb_aqi_dataframe(pivot)
    df = pivot.copy()
    df['AQI'] = aqi_df['AQI']
    df['AQI_Category'] = aqi_df['AQI_Category']
    
    # 1. Pollutant Distributions
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for idx, p in enumerate(TARGET_POLLUTANTS):
        sns.histplot(df[p].dropna(), kde=True, ax=axes[idx], color='#1f77b4', bins=40)
        axes[idx].set_title(f"Distribution of {p}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(f"{p} Concentration", fontsize=10)
        axes[idx].set_ylabel("Frequency", fontsize=10)
        
    # 8th plot: AQI distribution
    sns.histplot(df['AQI'].dropna(), kde=True, ax=axes[7], color='#d62728', bins=40)
    axes[7].set_title("Distribution of CPCB AQI", fontsize=12, fontweight='bold')
    axes[7].set_xlabel("AQI Value", fontsize=10)
    axes[7].set_ylabel("Frequency", fontsize=10)
    axes[8].axis('off')
    
    fig.suptitle("AeroCast — Pollutant and AQI Distributions (2023–2024)", fontsize=15, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(EDA_DIR, "01_pollutant_distributions.png"), dpi=300)
    plt.close(fig)
    print("Saved 01_pollutant_distributions.png")

    # 2. PM2.5 and PM10 Trends Over Time (Resampled daily for visual clarity)
    df_daily = df[TARGET_POLLUTANTS + ['AQI']].resample('D').mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_daily.index, df_daily['PM2.5'], label='PM2.5 (24h Daily Avg)', color='#d62728', alpha=0.85)
    ax.plot(df_daily.index, df_daily['PM10'], label='PM10 (24h Daily Avg)', color='#1f77b4', alpha=0.75)
    ax.axhline(60, color='#d62728', linestyle=':', label='PM2.5 CPCB Standard (60 µg/m³)')
    ax.axhline(100, color='#1f77b4', linestyle=':', label='PM10 CPCB Standard (100 µg/m³)')
    ax.set_title("Particulate Matter (PM2.5 & PM10) Historical Timeline (Delhi 2023–2024)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Concentration (µg/m³)", fontsize=11)
    ax.legend(frameon=True, loc='upper right')
    fig.savefig(os.path.join(EDA_DIR, "02_pm25_pm10_trends.png"), dpi=300)
    plt.close(fig)
    print("Saved 02_pm25_pm10_trends.png")

    # 3. Gaseous Pollutants Trends (NO2, SO2, CO, Ozone)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    axes[0, 0].plot(df_daily.index, df_daily['NO2'], color='#ff7f0e')
    axes[0, 0].set_title("Nitrogen Dioxide (NO2)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("µg/m³")
    
    axes[0, 1].plot(df_daily.index, df_daily['SO2'], color='#2ca02c')
    axes[0, 1].set_title("Sulfur Dioxide (SO2)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("µg/m³")
    
    axes[1, 0].plot(df_daily.index, df_daily['CO'], color='#9467bd')
    axes[1, 0].set_title("Carbon Monoxide (CO)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("mg/m³")
    axes[1, 0].set_xlabel("Date")
    
    axes[1, 1].plot(df_daily.index, df_daily['Ozone'], color='#17becf')
    axes[1, 1].set_title("Ozone (O3)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("µg/m³")
    axes[1, 1].set_xlabel("Date")
    
    fig.suptitle("Gaseous Pollutant Historical Profiles (Delhi 2023–2024)", fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(EDA_DIR, "03_gaseous_pollutants_trends.png"), dpi=300)
    plt.close(fig)
    print("Saved 03_gaseous_pollutants_trends.png")

    # 4. AQI Distribution & Timeline
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    cat_order = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    cat_counts = df['AQI_Category'].value_counts().reindex(cat_order).dropna()
    sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax1, palette='Spectral_r')
    ax1.set_title("CPCB AQI Category Frequency (Hours)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("CPCB Health Band")
    ax1.set_ylabel("Total Hours")
    for p in ax1.patches:
        ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                     ha='center', va='center', color='black', fontweight='bold')
                     
    ax2.plot(df_daily.index, df_daily['AQI'], color='#8c564b', linewidth=1.5)
    ax2.axhline(200, color='orange', linestyle='--', label='Poor Threshold (200)')
    ax2.axhline(300, color='red', linestyle='--', label='Very Poor Threshold (300)')
    ax2.axhline(400, color='darkred', linestyle='--', label='Severe Threshold (400)')
    ax2.set_title("CPCB Daily Average AQI Timeline", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("AQI Index Value")
    ax2.legend(frameon=True, loc='upper right')
    
    fig.savefig(os.path.join(EDA_DIR, "04_aqi_distribution_and_timeline.png"), dpi=300)
    plt.close(fig)
    print("Saved 04_aqi_distribution_and_timeline.png")

    # 5. Monthly Pollution Trends
    df['Month_Name'] = df.index.strftime('%b')
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_mean = df.groupby('Month_Name')[['PM2.5', 'PM10', 'AQI']].mean().reindex(month_order)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_mean.plot(kind='bar', ax=ax, width=0.8, colormap='tab10')
    ax.set_title("Seasonal/Monthly Variation in Major Pollutants & AQI (Delhi)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month of Year", fontsize=11)
    ax.set_ylabel("Mean Value", fontsize=11)
    ax.legend(frameon=True)
    fig.savefig(os.path.join(EDA_DIR, "05_monthly_pollution_trends.png"), dpi=300)
    plt.close(fig)
    print("Saved 05_monthly_pollution_trends.png")

    # 6. Hour-of-Day Diurnal Patterns
    df['Hour'] = df.index.hour
    diurnal = df.groupby('Hour')[['PM2.5', 'PM10', 'NO2', 'CO', 'Ozone']].mean()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ['PM2.5', 'PM10', 'NO2', 'Ozone']:
        ax.plot(diurnal.index, diurnal[col], marker='o', label=col, linewidth=2)
    ax.set_title("Diurnal (Hour-of-Day) Patterns across Pollutants", fontsize=13, fontweight='bold')
    ax.set_xlabel("Hour of Day (0–23 IST)", fontsize=11)
    ax.set_ylabel("Concentration (µg/m³)", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.legend(frameon=True)
    fig.savefig(os.path.join(EDA_DIR, "06_diurnal_hourly_patterns.png"), dpi=300)
    plt.close(fig)
    print("Saved 06_diurnal_hourly_patterns.png")

    # 7. Correlation Heatmap
    corr_cols = TARGET_POLLUTANTS + ['AQI']
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Pollutant and CPCB AQI Cross-Correlation Matrix", fontsize=13, fontweight='bold', pad=12)
    fig.savefig(os.path.join(EDA_DIR, "07_correlation_heatmap.png"), dpi=300)
    plt.close(fig)
    print("Saved 07_correlation_heatmap.png")

    # 8. Boxplots for Pollutant Distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, p in enumerate(TARGET_POLLUTANTS):
        sns.boxplot(y=df[p], ax=axes[idx], color='#6baed6')
        axes[idx].set_title(f"Boxplot: {p}", fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(f"{p} Reading")
    sns.boxplot(y=df['AQI'], ax=axes[7], color='#fb6a4a')
    axes[7].set_title("Boxplot: CPCB AQI", fontsize=12, fontweight='bold')
    axes[7].set_ylabel("AQI Value")
    
    fig.suptitle("Outlier and Spread Assessment across Pollutant Indicators", fontsize=14, fontweight='bold', y=1.02)
    fig.savefig(os.path.join(EDA_DIR, "08_pollutant_boxplots.png"), dpi=300)
    plt.close(fig)
    print("Saved 08_pollutant_boxplots.png")

    # 9. Missing-Value Analysis
    missing_pct = (df[TARGET_POLLUTANTS + ['AQI']].isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=missing_pct.index, y=missing_pct.values, ax=ax, palette='Blues_r')
    ax.set_title("Missing Value Percentage per Parameter at Shadipur (site_113)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Parameter Name", fontsize=11)
    ax.set_ylabel("Missing Percentage (%)", fontsize=11)
    ax.set_ylim(0, 10)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 0.2),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    fig.savefig(os.path.join(EDA_DIR, "09_missing_value_analysis.png"), dpi=300)
    plt.close(fig)
    print("Saved 09_missing_value_analysis.png")

    # 10. Outlier Percentile Analysis
    p_df = df[TARGET_POLLUTANTS].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    fig, ax = plt.subplots(figsize=(12, 5))
    p_summary = p_df[['min', '50%', '99%', 'max']]
    p_summary.plot(kind='bar', logy=True, ax=ax, colormap='viridis')
    ax.set_title("Pollutant Concentration Percentiles (Log Scale)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Pollutant", fontsize=11)
    ax.set_ylabel("Concentration (Log Scale)", fontsize=11)
    ax.legend(title="Statistic", frameon=True)
    fig.savefig(os.path.join(EDA_DIR, "10_outlier_percentile_analysis.png"), dpi=300)
    plt.close(fig)
    print("Saved 10_outlier_percentile_analysis.png")

    # Summary Tables
    summary_stats = df[TARGET_POLLUTANTS + ['AQI']].describe().T
    summary_stats.to_csv(os.path.join(EDA_DIR, "dataset_summary_statistics.csv"))
    
    completeness = pd.DataFrame({
        "Total_Expected_Hours": len(df),
        "Recorded_Hours": df[TARGET_POLLUTANTS + ['AQI']].count(),
        "Missing_Hours": df[TARGET_POLLUTANTS + ['AQI']].isna().sum(),
        "Missing_Percentage": df[TARGET_POLLUTANTS + ['AQI']].isna().mean() * 100
    })
    completeness.to_csv(os.path.join(EDA_DIR, "pollutant_completeness_summary.csv"))
    
    cat_df = df['AQI_Category'].value_counts(dropna=False).reset_index()
    cat_df.columns = ['CPCB_Category', 'Count']
    cat_df['Percentage'] = (cat_df['Count'] / len(df)) * 100
    cat_df.to_csv(os.path.join(EDA_DIR, "cpcb_aqi_category_distribution.csv"), index=False)
    
    print("\nAll 10 EDA charts and summary CSVs successfully generated in reports/eda/!")

if __name__ == '__main__':
    generate_eda_figures()

