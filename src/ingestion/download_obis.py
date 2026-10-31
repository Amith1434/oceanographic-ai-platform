
import requests
import json
import pandas as pd
import time
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API = "https://api.obis.org/v3/occurrence"

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

PAGE_SIZE = 10000

SPECIES = {
    "Blacktip Reef Shark": "Carcharhinus melanopterus",
    "Whale Shark": "Rhincodon typus",
    "Indian Threadfin Bream": "Setipinna tenuifilis"
}

OUTPUT_DIR = Path("data/raw/obis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load region configuration
with open("config/regions.json", "r") as f:
    config = json.load(f)

regions = config["regions"]

session = requests.Session()
session.headers.update({
    "User-Agent": "Oceanographic-AI-Platform/1.0"
})


# --------------------------------------------------
# DOWNLOAD FUNCTION
# --------------------------------------------------

def download_species(
    common_name,
    scientific_name,
    region_name,
    polygon
):

    print(f"\nDownloading {common_name} - {region_name}")

    params = {
        "scientificname": scientific_name,
        "geometry": polygon,
        "startdate": START_DATE,
        "enddate": END_DATE,
        "size": PAGE_SIZE
    }

    all_records = []
    after = None
    page = 1

    while True:

        if after:
            params["after"] = after

        response = session.get(
            API,
            params=params,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()
        records = data.get("results", [])

        if not records:
            break

        all_records.extend(records)

        print(
            f"Page {page}: {len(records)} records | "
            f"Total downloaded: {len(all_records)}"
        )

        # Stop if last page
        if len(records) < PAGE_SIZE:
            break

        # Cursor pagination
        last_id = records[-1].get("id")

        if not last_id:
            raise RuntimeError(
                "Missing record ID; cannot safely continue pagination."
            )

        after = last_id
        page += 1

        # Avoid excessive API requests
        time.sleep(0.5)

    print(
        f"Completed: {common_name} - {region_name}: "
        f"{len(all_records)} records"
    )

    return all_records


# --------------------------------------------------
# MAIN DOWNLOAD PIPELINE
# --------------------------------------------------

all_data = []

for region_key, region in regions.items():

    polygon = (
        f"POLYGON(({region['min_lon']} {region['min_lat']},"
        f"{region['max_lon']} {region['min_lat']},"
        f"{region['max_lon']} {region['max_lat']},"
        f"{region['min_lon']} {region['max_lat']},"
        f"{region['min_lon']} {region['min_lat']}))"
    )

    for common_name, scientific_name in SPECIES.items():

        records = download_species(
            common_name,
            scientific_name,
            region["name"],
            polygon
        )

        for record in records:

            record["common_name"] = common_name
            record["query_region"] = region["name"]

        all_data.extend(records)


# --------------------------------------------------
# SAVE RAW DATA
# --------------------------------------------------

if not all_data:
    print("\nNo records downloaded.")
else:

    df = pd.DataFrame(all_data)

    # Remove duplicate OBIS record IDs
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])

    output_file = OUTPUT_DIR / "obis_occurrences_2015_2025.csv"

    df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print("OBIS DOWNLOAD COMPLETED")
    print("=" * 60)

    print("Total unique records:", len(df))
    print("Species:", df["common_name"].nunique())
    print("Regions:", df["query_region"].nunique())
    print("Columns:", len(df.columns))

    print("\nRecords by species and region:")

    print(
        df.groupby(
            ["query_region", "common_name"]
        ).size().to_string()
    )

    print("\nSaved to:", output_file)
