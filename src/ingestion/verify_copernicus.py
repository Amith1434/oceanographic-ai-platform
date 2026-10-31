
import xarray as xr
from pathlib import Path

regions = ["arabian_sea", "bay_of_bengal"]
years = range(2019, 2026)

base_dir = Path("data/raw/copernicus")

expected_variables = {"thetao", "so", "uo", "vo"}

all_passed = True

for region in regions:

    print(f"\n{'=' * 60}")
    print(f"REGION: {region.upper()}")
    print(f"{'=' * 60}")

    for year in years:

        file_path = base_dir / region / f"{year}.nc"

        print(f"\nChecking {year}...")

        if not file_path.exists():
            print("FAILED: File does not exist")
            all_passed = False
            continue

        try:
            with xr.open_dataset(file_path) as ds:

                # Check variables
                actual_variables = set(ds.data_vars)
                missing_variables = expected_variables - actual_variables

                # Check monthly records
                dates = ds.time.values
                months = len(dates)
                correct_year = all(
                    date_year == year
                    for date_year in ds.time.dt.year.values
                )

                # Check spatial coordinates
                lat_min = float(ds.latitude.min())
                lat_max = float(ds.latitude.max())
                lon_min = float(ds.longitude.min())
                lon_max = float(ds.longitude.max())

                # Check depth
                depths = ds.depth.values

                # Check missing values
                valid_percentages = {}

                for variable in expected_variables:
                    if variable in ds:
                        data = ds[variable]
                        valid = data.count().item()
                        total = data.size

                        valid_percentages[variable] = round(
                            valid / total * 100, 2
                        )

                passed = (
                    not missing_variables
                    and months == 12
                    and correct_year
                    and all(
                        value > 0
                        for value in valid_percentages.values()
                    )
                )

                print("Variables:", sorted(actual_variables))
                print("Monthly records:", months)
                print("Correct year:", correct_year)
                print("Date range:", dates[0], "to", dates[-1])
                print("Latitude:", lat_min, "to", lat_max)
                print("Longitude:", lon_min, "to", lon_max)
                print("Depth levels:", depths)
                print("Valid data (%):", valid_percentages)

                if passed:
                    print("STATUS: PASSED")
                else:
                    print("STATUS: FAILED")
                    all_passed = False

        except Exception as error:
            print("STATUS: FAILED")
            print("Error:", error)
            all_passed = False

print("\n" + "=" * 60)

if all_passed:
    print("ALL 14 COPERNICUS FILES PASSED BASIC VERIFICATION")
else:
    print("SOME FILES NEED INVESTIGATION")

print("=" * 60)
