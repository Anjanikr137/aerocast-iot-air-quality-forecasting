"""
Test downloading a full month for site_113.
"""

import os
import requests
import time

api_key = os.environ.get("XKDR_API_KEY", "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe")
headers = {"Authorization": f"Bearer {api_key}"}

# Query 1 full month: 2024-01-01 to 2024-01-31
url = "https://airquality.xkdr.org/v1/measurements"
params = {
    "station": "site_113",
    "start": "2024-01-01",
    "end": "2024-01-31",
    "format": "json"
}

t0 = time.time()
resp = requests.get(url, headers=headers, params=params, timeout=60)
elapsed = time.time() - t0
print(f"Status: {resp.status_code}, time: {elapsed:.2f}s")
if resp.status_code == 200:
    res = resp.json()
    meta = res.get("meta", {})
    data = res.get("data", [])
    print("Meta:", meta)
    print(f"Data rows: {len(data)}, truncated: {meta.get('truncated')}")

