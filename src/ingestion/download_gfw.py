
import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_TOKEN = os.getenv("GFW_API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("GFW_API_TOKEN not found in .env")

API_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

DATASET = "public-global-fishing-effort:latest"

# --------------------------------------------------
# 2. Test configuration
# --------------------------------------------------

REGION_NAME = "bay_of_bengal"

# Small test polygon: 50-51 E, 10-11 N
geojson = {
    "type": "Polygon",
    "coordinates": [[
        [78, 0],
        [100, 0],
        [100, 23],
        [78, 23],
        [78, 0]
    ]]
}

START_DATE = "2021-10-01"
END_DATE = "2021-11-01"

# --------------------------------------------------
# 3. API parameters
# --------------------------------------------------

params = {
    "datasets[0]": DATASET,
    "date-range": "2021-10-01,2021-11-01",
    "temporal-resolution": "MONTHLY",
    "spatial-resolution": "LOW",
    "group-by": "FLAG",
    "format": "JSON"
}
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# --------------------------------------------------
# 4. Send request
# --------------------------------------------------

print("Requesting GFW fishing activity...")
print("Region:", REGION_NAME)
print("Period:", START_DATE, "to", END_DATE)

try:
    response = requests.post(
        API_URL,
        params=params,
        headers=headers,
        json={"geojson": geojson},
        timeout=120
    )

    print("HTTP status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    # --------------------------------------------------
    # 5. Extract dataset records
    # --------------------------------------------------

    entries = result.get("entries", [])

    records = []

    for entry in entries:
        for dataset_key, dataset_records in entry.items():
            if dataset_key.startswith("public-global-fishing-effort"):
                if dataset_records:
                    records.extend(dataset_records)

    print("Total API entries:", len(entries))
    print("Fishing activity records:", len(records))

    # --------------------------------------------------
    # 6. Save raw response
    # --------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "gfw"
        / "test"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "bay_of_bengal_oct2021_test.json"
    output_file.write_text(
        json.dumps(result, indent=4),
        encoding="utf-8"
    )

    print("Raw response saved:", output_file)

    # --------------------------------------------------
    # 7. Preview records
    # --------------------------------------------------

    if records:
        print("\nFirst five records:")

        for record in records[:5]:
            print(record)

    else:
        print("\nNo fishing activity records returned.")

except requests.exceptions.HTTPError:
    print("GFW API returned an HTTP error.")
    print(response.text[:2000])

except requests.exceptions.RequestException as error:
    print("Network/API request failed:", error)

except ValueError:
    print("Could not decode the API response as JSON.")
    print(response.text[:1000])
