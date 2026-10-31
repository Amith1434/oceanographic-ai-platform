
import pandas as pd

file = "data/raw/obis/obis_occurrences_2015_2025.csv"

df = pd.read_csv(file, low_memory=False)

# Inspect date-related columns
date_columns = [
    col for col in df.columns
    if "date" in col.lower()
]

print("Date-related columns:")
print(date_columns)

print("\nRecords by species:")

for species, group in df.groupby("common_name"):

    print("\n" + "=" * 55)
    print(species)
    print("Total records:", len(group))

    for col in date_columns:

        if col not in group.columns:
            continue

        valid = group[col].notna().sum()

        print(f"{col}: {valid}/{len(group)} non-missing")

    print("\nSample date values:")

    for col in date_columns:
        if col in group.columns:
            print(f"\n{col}:")
            print(group[col].dropna().head(5).to_list())
