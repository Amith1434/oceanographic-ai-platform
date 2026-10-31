
import os
import json
import time
from pathlib import Path
from datetime import date

import requests
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TOKEN = os.getenv("GFW_API_TOKEN")

if not TOKEN:
    raise RuntimeError("GFW_API_TOKEN not found in .env")

API_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
DATASET = "public-global-fishing-effort:latest"

START_YEAR = 2019
END_YEAR = 2025

REGIONS = {
    "arabian_sea": {
        "coordinates": [[
            [50, 0], [78, 0], [78, 25],
            [50, 25], [50, 0]
        ]]
    },
    "bay_of_bengal": {
        "coordinates": [[
            [78, 0], [100, 0], [100, 23],
            [78, 23], [78, 0]
        ]]
    }
}

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

session = requests.Session()
session.headers.update(HEADERS)


# --------------------------------------------------
# 2. Generate monthly date ranges
# --------------------------------------------------

def generate_months():
    months = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = date(year, month, 1)

            if month == 12:
                end = date(year + 1, 1, 1)
            else:
                end = date(year, month + 1, 1)

            months.append((
                start.isoformat(),
                end.isoformat(),
                f"{year}-{month:02d}"
            ))

    return months


# --------------------------------------------------
# 3. Extract fishing records
# --------------------------------------------------

def extract_records(data):
    records = []

    for entry in data.get("entries", []):
        for key, values in entry.items():
            if key.startswith("public-global-fishing-effort"):
                if values:
                    records.extend(values)

    return records


# --------------------------------------------------
# 4. Make API request with retries
# --------------------------------------------------

def fetch_month(region_name, region, start, end):

    params = {
        "datasets[0]": DATASET,
        "date-range": f"{start},{end}",
        "temporal-resolution": "MONTHLY",
        "spatial-resolution": "LOW",
        "group-by": "FLAG",
        "format": "JSON"
    }

    geojson = {
        "type": "Polygon",
        "coordinates": region["coordinates"]
    }

    for attempt in range(1, 4):
        try:
            response = session.post(
                API_URL,
                params=params,
                json={"geojson": geojson},
                timeout=180
            )

            if response.status_code == 429 or response.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"Temporary API error: HTTP {response.status_code}",
                    response=response
                )

            response.raise_for_status()
            return response.json()

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError
        ) as error:

            print(f"Request attempt {attempt}/3 failed: {error}")

            if attempt == 3:
                raise

            time.sleep(5 * attempt)


# --------------------------------------------------
# 5. Download one region
# --------------------------------------------------

def download_region(region_name, region):

    print("\n" + "=" * 55)
    print("REGION:", region_name)
    print("=" * 55)

    raw_dir = ROOT / "data" / "raw" / "gfw" / region_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = ROOT / "data" / "processed" / "gfw" / region_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    months = generate_months()

    for index, (start, end, month) in enumerate(months, 1):

        output_file = raw_dir / f"{month}.json"

        # Resume: skip previously downloaded months
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)

                if isinstance(cached.get("entries"), list):
                    print(
                        f"[{index}/{len(months)}] "
                        f"{month}: already downloaded, skipping"
                    )
                    continue

            except (json.JSONDecodeError, OSError):
                print(f"{month}: invalid cached file, downloading again")

        print(f"\n[{index}/{len(months)}] Downloading {month}...")

        data = fetch_month(
            region_name, region, start, end
        )

        # Save original API response
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f)

        records = extract_records(data)

        print("Records received:", len(records))
        print("Raw file saved:", output_file.name)

        # Sequential requests; avoid overwhelming the API
        time.sleep(1)

    # --------------------------------------------------
    # 6. Build consolidated CSV from all monthly files
    # --------------------------------------------------

    all_records = []

    for start, end, month in months:

        file_path = raw_dir / f"{month}.json"

        if not file_path.exists():
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = extract_records(data)

        for record in records:
            row = record.copy()
            row["region"] = region_name
            all_records.append(row)

    df = pd.DataFrame(all_records)

    csv_path = (
        processed_dir / f"gfw_{START_YEAR}_{END_YEAR}.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\nREGION DOWNLOAD COMPLETE:", region_name)
    print("Total records:", len(df))
    print("CSV saved:", csv_path)

    if not df.empty:
        print("Missing values:")
        print(df.isnull().sum())

        print("\nTotal apparent fishing hours:")
        print(df["hours"].sum())


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    print("GLOBAL FISHING WATCH DATA COLLECTION")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print("Regions:", len(REGIONS))
    print("Monthly requests per region:", len(generate_months()))

    for region_name, region in REGIONS.items():
        download_region(region_name, region)

    print("\nALL GFW DOWNLOADS FINISHED.")
