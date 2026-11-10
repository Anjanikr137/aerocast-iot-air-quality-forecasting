"""
Analyzes the retrieved Delhi stations to find the best candidate stations
with comprehensive coverage of PM2.5, PM10, NO2, SO2, CO, Ozone, and NH3.
"""

import json
import ast
import pandas as pd

def main():
    df = pd.read_csv('reports/eda/delhi_stations_raw.csv')
    print(f"Total stations: {len(df)}")
    
    # Sort by n_rows descending
    df_sorted = df.sort_values(by='n_rows', ascending=False).reset_index(drop=True)
    
    print("\nTop 15 Delhi stations by row count:")
    cols_to_show = ['station_id', 'station_name', 'first_seen', 'last_seen', 'n_rows']
    print(df_sorted[cols_to_show].head(15).to_string())

    required_pollutants = {'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone'}
    
    def parse_params(p):
        if pd.isna(p):
            return []
        try:
            return ast.literal_eval(p)
        except Exception:
            return []

    df_sorted['param_list'] = df_sorted['parameters'].apply(parse_params)
    df_sorted['param_count'] = df_sorted['param_list'].apply(len)
    df_sorted['has_all_6'] = df_sorted['param_list'].apply(lambda s: required_pollutants.issubset(set(s)))
    df_sorted['has_nh3'] = df_sorted['param_list'].apply(lambda s: 'NH3' in s)

    print("\n=== Stations with ALL 6 Core Pollutants + NH3 ===")
    candidates = df_sorted[df_sorted['has_all_6']].copy()
    print(candidates[['station_id', 'station_name', 'first_seen', 'last_seen', 'n_rows', 'has_nh3']].to_string())
    
    # Save candidates
    candidates.to_csv('reports/eda/delhi_candidate_stations.csv', index=False)

if __name__ == "__main__":
    main()

