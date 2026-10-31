
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load the project-level .env file
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

API_TOKEN = os.getenv("GFW_API_TOKEN")

if not API_TOKEN:
    raise RuntimeError("GFW_API_TOKEN was not found in .env")

URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"

# Small test area inside the Arabian Sea
geojson = {
    "type": "Polygon",
    "coordinates": [[
        [50, 10],
        [51, 10],
        [51, 11],
        [50, 11],
        [50, 10]
    ]]
}

params = {
    "datasets[0]": "public-global-fishing-effort:latest",
    "date-range": "2021-01-01,2022-01-01",
    "temporal-resolution": "YEARLY",
    "spatial-resolution": "LOW",
    "group-by": "FLAG",
    "format": "JSON"
}

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

print("Sending GFW API test request...")

try:
    response = requests.post(
        URL,
        params=params,
        headers=headers,
        json={"geojson": geojson},
        timeout=120
    )

    print("HTTP status:", response.status_code)
    print("Content type:", response.headers.get("Content-Type"))
    print("Response size:", len(response.content), "bytes")

    if response.status_code == 200:
        print("GFW API request successful!")

        # Save the test response without exposing the token
        output_dir = PROJECT_ROOT / "data" / "raw" / "gfw" / "test"
        output_dir.mkdir(parents=True, exist_ok=True)

        content_type = response.headers.get("Content-Type", "").lower()

        if "json" in content_type:
            output_file = output_dir / "arabian_sea_2021_test.json"
            output_file.write_text(
                response.text,
                encoding="utf-8"
            )
        else:
            output_file = output_dir / "arabian_sea_jan2019_test_response.bin"
            output_file.write_bytes(response.content)

        print("Saved response to:", output_file)

    else:
        print("GFW API returned an error.")
        print(response.text[:1500])

except requests.exceptions.RequestException as error:
    print("Request failed:", error)
