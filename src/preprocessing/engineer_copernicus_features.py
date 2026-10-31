
import xarray as xr
import numpy as np
from pathlib import Path

regions = ["arabian_sea", "bay_of_bengal"]

base_dir = Path("data/processed/copernicus")

for region in regions:

    print("\n" + "=" * 55)
    print(f"FEATURE ENGINEERING: {region.upper()}")
    print("=" * 55)

    input_file = (
        base_dir / region / "copernicus_2019_2025.nc"
    )

    output_file = (
        base_dir / region / "copernicus_features_2019_2025.nc"
    )

    print("Loading dataset...")

    with xr.open_dataset(input_file) as source:

        # Load data into memory before closing source
        ds = source.load()

    # 1. Calculate current speed
    print("Calculating current speed...")

    ds["current_speed"] = np.sqrt(
        ds["uo"] ** 2 + ds["vo"] ** 2
    )

    # 2. Calculate current direction
    print("Calculating current direction...")

    direction = (
        np.degrees(np.arctan2(ds["vo"], ds["uo"])) + 360
    ) % 360

    ds["current_direction"] = direction.where(
        ds["current_speed"] > 0
    )

    # Add descriptions
    ds["current_speed"].attrs = {
        "long_name": "Ocean current speed",
        "units": "m/s"
    }

    ds["current_direction"].attrs = {
        "long_name": "Ocean current direction",
        "units": "degrees counterclockwise from east"
    }

    # 3. Print feature information
    print("\nAVAILABLE VARIABLES:")
    print(list(ds.data_vars))

    print("\nFEATURE SUMMARY:")

    for variable in [
        "thetao",
        "so",
        "uo",
        "vo",
        "current_speed",
        "current_direction"
    ]:

        print(f"\n{variable}")
        print("Minimum:", float(ds[variable].min()))
        print("Maximum:", float(ds[variable].max()))
        print(
            "Missing values:",
            int(ds[variable].isnull().sum())
        )

    # 4. Save engineered dataset
    print("\nSaving engineered dataset...")

    ds.to_netcdf(output_file)

    print("Saved:", output_file)

    ds.close()

print("\n" + "=" * 55)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 55)
