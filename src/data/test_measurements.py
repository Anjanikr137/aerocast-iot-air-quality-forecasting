"""
Test querying /v1/measurements endpoint for Shadipur (site_113)
to inspect the exact API schema and response format.
"""

import os
import requests
import json

api_key = os.environ.get("XKDR_API_KEY", "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe")
headers = {"Authorization": f"Bearer {api_key}"}

# Query a 3-day sample
url = "https://airquality.xkdr.org/v1/measurements"
params = {
    "station": "site_113",
    "start": "2024-01-01",
    "end": "2024-01-03",
    "format": "json"
}

print(f"Requesting {url} with params {params}...")
try:
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    print("Status code:", resp.status_code)
    data = resp.json()
    if isinstance(data, list):
        print(f"Returned list with {len(data)} items")
        if data:
            print("Sample record:", json.dumps(data[0], indent=2))
    elif isinstance(data, dict):
        print("Returned dict with keys:", list(data.keys()))
        for k in data.keys():
            val = data[k]
            if isinstance(val, list):
                print(f"Key '{k}' has {len(val)} items")
                if val:
                    print(f"First item of '{k}':", json.dumps(val[0], indent=2))
            else:
                print(f"Key '{k}':", val)
except Exception as e:
    print("Error:", e)

