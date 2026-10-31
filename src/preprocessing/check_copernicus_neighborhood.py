
import xarray as xr
from pathlib import Path

checks = {
    "arabian_sea": {
        "variable": "so",
        "lat": 22.25,
        "lon": 72.5,
        "date": "2021-10-01"
    },
    "bay_of_bengal": {
        "variable": "so",
        "lat": 22.166666,
        "lon": 90.666667,
        "date": "2025-10-01"
    }
}

base_dir = Path("data/processed/copernicus")

for region, info in checks.items():

    print("\n" + "=" * 55)
    print(region.upper())
    print("=" * 55)

    file_path = (
        base_dir / region /
        "copernicus_features_2019_2025.nc"
    )

    with xr.open_dataset(file_path) as ds:

        # Select the relevant date and surface depth
        selected = ds.sel(
            time=info["date"],
            depth=0.494025,
            method="nearest"
        )

        # Select a 5x5 neighbourhood
        
        # Select a 5x5 neighbourhood
        neighborhood = selected[info["variable"]].sel(
            latitude=slice(
                info["lat"] - 0.17,
                info["lat"] + 0.17
            ),
            longitude=slice(
                info["lon"] - 0.17,
                info["lon"] + 0.17
            )
        )

        print("\nNearby salinity values:")
        print(neighborhood.to_pandas())

        # Check whether valid values exist
        valid_count = int(neighborhood.count())

        print("\nValid neighbouring cells:", valid_count)

        if valid_count == 0:
            print("No valid ocean measurements in this neighbourhood.")
            continue

        print("\nNeighbourhood statistics:")
        print("Minimum:", float(neighborhood.min(skipna=True)))
        print("Maximum:", float(neighborhood.max(skipna=True)))
        print("Mean:", float(neighborhood.mean(skipna=True)))
        print("Median:", float(neighborhood.median(skipna=True)))
