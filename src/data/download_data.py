"""
Data Ingestion Module for AeroCast.
Downloads historical air-quality observations from the XKDR Air Quality Database
for selected CPCB monitoring station (site_113 - Shadipur, Delhi).
Preserves raw data without modification, verifies no truncation occurs,
and logs the download manifest.
"""

import os
import sys
import json
import time
import calendar
import requests
import pandas as pd
from datetime import datetime

API_BASE = "https://airquality.xkdr.org"

def get_api_key():
    key = os.environ.get("XKDR_API_KEY")
    if not key:
        key = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"
    return key

def download_batch(station_id, start_date, end_date, api_key):
    url = f"{API_BASE}/v1/measurements"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "station": station_id,
        "start": start_date,
        "end": end_date,
        "format": "json"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 200:
                res = resp.json()
                meta = res.get("meta", {})
                data = res.get("data", [])
                
                # Check for truncation
                if meta.get("truncated", False):
                    print(f"WARNING: Batch {start_date} to {end_date} was truncated! Splitting into two halves...")
                    # Split in half
                    d_start = datetime.strptime(start_date, "%Y-%m-%d")
                    d_end = datetime.strptime(end_date, "%Y-%m-%d")
                    mid_day = d_start + (d_end - d_start) / 2
                    mid_str = mid_day.strftime("%Y-%m-%d")
                    next_str = (mid_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    data1, meta1 = download_batch(station_id, start_date, mid_str, api_key)
                    data2, meta2 = download_batch(station_id, next_str, end_date, api_key)
                    return data1 + data2, {"split": True, "part1": meta1, "part2": meta2}
                    
                return data, meta
            else:
                print(f"Attempt {attempt+1}: Status {resp.status_code} for {start_date} to {end_date}: {resp.text[:100]}")
                time.sleep(2)
        except Exception as e:
            print(f"Attempt {attempt+1} exception: {e}")
            time.sleep(2)
            
    raise RuntimeError(f"Failed to download batch {start_date} to {end_date} after {max_retries} attempts.")

def download_station_data(station_id="site_113", start_year=2023, end_year=2024):
    api_key = get_api_key()
    raw_dir = "data/raw"
    batch_dir = os.path.join(raw_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    all_records = []
    manifest = {
        "station_id": station_id,
        "station_name": "Shadipur, Delhi - CPCB",
        "download_timestamp": datetime.utcnow().isoformat(),
        "batches": []
    }
    
    print(f"Starting download for {station_id} ({manifest['station_name']}) from {start_year} to {end_year}...")
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Calculate start and end date of the month
            _, last_day = calendar.monthrange(year, month)
            start_date = f"{year:04d}-{month:02d}-01"
            end_date = f"{year:04d}-{month:02d}-{last_day:02d}"
            
            batch_filename = f"{station_id}_{year:04d}_{month:02d}.json"
            batch_path = os.path.join(batch_dir, batch_filename)
            
            # Check if cached locally to avoid re-downloading if already done
            if os.path.exists(batch_path):
                print(f"Loading cached batch {year:04d}-{month:02d} from {batch_filename}...")
                with open(batch_path, "r", encoding="utf-8") as f:
                    batch_info = json.load(f)
                    records = batch_info.get("data", [])
                    meta = batch_info.get("meta", {})
            else:
                print(f"Downloading batch {year:04d}-{month:02d} ({start_date} to {end_date})...")
                records, meta = download_batch(station_id, start_date, end_date, api_key)
                
                with open(batch_path, "w", encoding="utf-8") as f:
                    json.dump({"meta": meta, "data": records}, f)
                time.sleep(0.5)  # respectful pacing
                
            all_records.extend(records)
            manifest["batches"].append({
                "period": f"{year:04d}-{month:02d}",
                "start_date": start_date,
                "end_date": end_date,
                "rows_retrieved": len(records),
                "truncated": meta.get("truncated", False)
            })
            print(f"  -> Retrieved {len(records)} records (Truncated: {meta.get('truncated', False)})")

    # Save manifest
    manifest["total_records"] = len(all_records)
    manifest_path = os.path.join(raw_dir, "download_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nDownload completed. Total raw records: {len(all_records)}")
    
    # Save combined raw data as CSV (raw unadulterated table)
    df_raw = pd.DataFrame(all_records)
    raw_csv_path = os.path.join(raw_dir, f"{station_id}_raw.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Saved complete raw dataset to {raw_csv_path} ({os.path.getsize(raw_csv_path) / (1024*1024):.2f} MB)")
    
    return raw_csv_path

if __name__ == "__main__":
    download_station_data()

