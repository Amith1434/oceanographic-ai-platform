
import xarray as xr
import numpy as np
from pathlib import Path

regions = ["arabian_sea", "bay_of_bengal"]

base_dir = Path("data/processed/copernicus")

for region in regions:

    print("\n" + "=" * 55)
    print(region.upper())
    print("=" * 55)

    file_path = (
        base_dir / region /
        "copernicus_features_2019_2025.nc"
    )

    with xr.open_dataset(file_path) as ds:

        for variable in ["thetao", "so", "current_speed"]:

            data = ds[variable]

            print(f"\n{variable}")

            # Statistical summary
            for percentile in [0.01, 0.05, 0.50, 0.95, 0.99]:

                value = float(
                    data.quantile(percentile, skipna=True)
                )

                print(
                    f"{percentile * 100:.0f}th percentile:",
                    value
                )

            # Convert to NumPy array
            values = data.values

            # Find valid minimum and maximum
            min_index = np.unravel_index(
                np.nanargmin(values),
                values.shape
            )

            max_index = np.unravel_index(
                np.nanargmax(values),
                values.shape
            )

            # Get coordinates of extreme values
            min_location = {
                dim: data.coords[dim].values[index].item()
                for dim, index in zip(data.dims, min_index)
            }

            max_location = {
                dim: data.coords[dim].values[index].item()
                for dim, index in zip(data.dims, max_index)
            }

            print("\nMinimum value:", np.nanmin(values))
            print("Minimum location:", min_location)

            print("\nMaximum value:", np.nanmax(values))
            print("Maximum location:", max_location)

print("\nOUTLIER INVESTIGATION COMPLETED")
