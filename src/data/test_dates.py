"""
Audit actual observation availability for top stations across years 2020-2025.
"""

import os
import requests
import json
import pandas as pd

api_key = os.environ.get("XKDR_API_KEY", "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe")
headers = {"Authorization": f"Bearer {api_key}"}

stations = ["site_113", "site_115", "site_114"]
test_dates = [
    ("2021-01-01", "2021-01-02"),
    ("2022-01-01", "2022-01-02"),
    ("2023-01-01", "2023-01-02"),
    ("2024-01-01", "2024-01-02"),
    ("2024-06-01", "2024-06-02"),
    ("2024-11-01", "2024-11-02"),
]

print("Testing query accessibility across years and stations...")
for st in stations:
    print(f"\n=== Station: {st} ===")
    for start, end in test_dates:
        url = "https://airquality.xkdr.org/v1/measurements"
        params = {"station": st, "start": start, "end": end, "format": "json"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            if resp.status_code == 200:
                res_json = resp.json()
                meta = res_json.get("meta", {})
                data = res_json.get("data", [])
                params_found = set(d["parameter_name"] for d in data) if data else set()
                print(f"[{start} to {end}] Status 200: {len(data)} rows, params: {len(params_found)} {list(params_found)[:5]}")
            else:
                print(f"[{start} to {end}] Status {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"[{start} to {end}] Error: {e}")

