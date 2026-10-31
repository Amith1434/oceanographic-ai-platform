
import xarray as xr
import json
from pathlib import Path

# Load project configuration
with open("config/regions.json", "r") as file:
    config = json.load(file)

regions = config["regions"]

start_year = 2019
end_year = 2025

raw_dir = Path("data/raw/copernicus")
processed_dir = Path("data/processed/copernicus")

variables = ["thetao", "so", "uo", "vo"]

# Create output directory
processed_dir.mkdir(parents=True, exist_ok=True)


def prepare_region(region_key, region_info):

    print("\n" + "=" * 60)
    print(f"PROCESSING: {region_info['name']}")
    print("=" * 60)

    region_raw_dir = raw_dir / region_key

    datasets = []

    try:
        # Load annual files
        for year in range(start_year, end_year + 1):

            file_path = region_raw_dir / f"{year}.nc"

            print(f"Loading {year}...")

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_path}"
                )

            ds = xr.open_dataset(file_path)

            # Keep only required variables
            ds = ds[variables]

            datasets.append(ds)

        # Combine all annual datasets along time
        print("\nCombining annual datasets...")

        combined = xr.concat(
            datasets,
            dim="time"
        )

        # Sort records chronologically
        combined = combined.sortby("time")

        # Check for duplicate timestamps
        timestamps = combined.time.values

        if len(timestamps) != len(set(timestamps)):
            raise ValueError("Duplicate timestamps detected!")

        # Display combined dataset information
        print("\nCOMBINED DATASET")
        print(combined)

        print("\nTIME RECORDS:", len(combined.time))
        print("VARIABLES:", list(combined.data_vars))
        print("DEPTH LEVELS:", combined.depth.values)

        # Missing value report
        print("\nMISSING VALUE REPORT")

        for variable in variables:

            total = combined[variable].size
            missing = combined[variable].isnull().sum().item()
            valid = total - missing

            print(f"\n{variable}")
            print("Total:", total)
            print("Valid:", valid)
            print("Missing:", missing)
            print(
                "Valid percentage:",
                round(valid / total * 100, 2),
                "%"
            )

        # Save combined dataset
        output_file = (
            processed_dir
            / region_key
        )

        output_file.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_file
            / "copernicus_2019_2025.nc"
        )

        print("\nSaving processed dataset...")

        combined.to_netcdf(output_path)

        print("Saved:", output_path)
        print("PROCESSING COMPLETED!")

        combined.close()

    finally:
        # Close all source datasets
        for ds in datasets:
            ds.close()


# Process both regions independently
for region_key, region_info in regions.items():

    prepare_region(region_key, region_info)


print("\n" + "=" * 60)
print("COPERNICUS PREPROCESSING COMPLETED")
print("=" * 60)
