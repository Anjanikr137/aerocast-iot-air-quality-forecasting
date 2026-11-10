"""
Station Discovery and Coverage Audit Module for XKDR Air Quality Database.
Discovers Delhi CPCB monitoring stations and evaluates historical coverage across pollutants.
"""

import os
import sys
import json
import requests
import pandas as pd

API_BASE = "https://airquality.xkdr.org"

def get_api_key():
    key = os.environ.get("XKDR_API_KEY")
    if not key:
        # Fallback to public demo key if not set in environment
        key = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
    return key

def fetch_delhi_stations():
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{API_BASE}/v1/stations?city=Delhi"
    
    print(f"Fetching stations from {url}...")
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error connecting to XKDR API: {e}")
        return []

    if isinstance(data, dict):
        stations = data.get("stations", data.get("data", data.get("results", [])))
    else:
        stations = data
    return stations

def main():
    stations = fetch_delhi_stations()
    print(f"Total Delhi stations retrieved: {len(stations)}")
    
    if not stations:
        print("No stations returned or API connection issue.")
        sys.exit(1)
        
    df_stations = pd.DataFrame(stations)
    os.makedirs("reports/eda", exist_ok=True)
    df_stations.to_csv("reports/eda/delhi_stations_raw.csv", index=False)
    
    print("\nColumns available:", df_stations.columns.tolist())
    print("\nSample stations:")
    for idx, row in df_stations.head(5).iterrows():
        print(f"- {row.get('station_id')}: {row.get('station_name')} (Source: {row.get('source')}, First: {row.get('first_reading')}, Last: {row.get('last_reading')})")

if __name__ == "__main__":
    main()

