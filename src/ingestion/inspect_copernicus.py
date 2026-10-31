import xarray as xr

file_path = "data/raw/copernicus/arabian_sea_january_2015.nc"

ds = xr.open_dataset(file_path)

print("\nDATASET DETAILS")
print(ds)

print("\nVARIABLES")
print(list(ds.data_vars))

print("\nCOORDINATES")
print(list(ds.coords))

print("\nMISSING VALUES")
for variable in ds.data_vars:
    print(variable, ds[variable].isnull().sum().item())

print("\nDEPTH COORDINATES")
print(ds["depth"].values)
print("\nVALID DATA ANALYSIS")

for variable in ds.data_vars:
    total = ds[variable].size
    missing = ds[variable].isnull().sum().item()
    valid = total - missing

    print(f"\n{variable}")
    print("Total grid points:", total)
    print("Valid values:", valid)
    print("Missing values:", missing)
    print("Valid percentage:", round((valid / total) * 100, 2), "%")
    print("Minimum:", ds[variable].min().item())
    print("Maximum:", ds[variable].max().item())
ds.close()