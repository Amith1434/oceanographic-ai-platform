
import copernicusmarine
import json
from pathlib import Path

# Load project configuration
with open("config/regions.json", "r") as file:
    config = json.load(file)

regions = config["regions"]
period = config["data_period"]

start_year = int(period["environment_start"][:4])
end_year = int(period["environment_end"][:4])

dataset_id = "cmems_mod_glo_phy_my_0.083deg_P1M-m"

variables = ["thetao", "so", "uo", "vo"]

base_dir = Path("data/raw/copernicus")

# Download region by region, year by year
for region_key, region in regions.items():

    print(f"\n{'='*50}")
    print(f"REGION: {region['name']}")
    print(f"{'='*50}")

    region_dir = base_dir / region_key
    region_dir.mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year + 1):

        output_file = region_dir / f"{year}.nc"

        if output_file.exists():
            print(f"Skipping {year}: File already exists")
            continue

        print(f"\nDownloading {region['name']} - {year}")

        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,

                minimum_longitude=region["min_lon"],
                maximum_longitude=region["max_lon"],
                minimum_latitude=region["min_lat"],
                maximum_latitude=region["max_lat"],

                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:59:59",

                minimum_depth=0.494,
                maximum_depth=5,

                output_filename=str(output_file)
            )

            print(f"Successfully downloaded: {output_file}")

        except Exception as error:
            print(f"FAILED: {region['name']} - {year}")
            print(error)

print("\nCopernicus download process finished.")
