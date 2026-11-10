"""
List all distinct parameters and units recorded for site_113 (Shadipur).
"""

import os
import requests
import pandas as pd

api_key = os.environ.get("XKDR_API_KEY", "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe")
headers = {"Authorization": f"Bearer {api_key}"}

url = "https://airquality.xkdr.org/v1/measurements"
params = {
    "station": "site_113",
    "start": "2024-01-01",
    "end": "2024-01-02",
    "format": "json"
}

resp = requests.get(url, headers=headers, params=params, timeout=20)
data = resp.json().get("data", [])
df = pd.DataFrame(data)

print("Unique parameters at site_113:")
summary = df.groupby(["parameter_name", "unit"]).size().reset_index(name="count")
print(summary.to_string(index=False))

