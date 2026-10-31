
import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/obis/obis_occurrences_2015_2025.csv")
OUTPUT_DIR = Path("data/processed/obis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, low_memory=False)

print("=" * 60)
print("INITIAL DATA")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# Inspect important fields
required_columns = [
    "id",
    "scientificName",
    "eventDate",
    "decimalLatitude",
    "decimalLongitude",
    "query_region"
]

print("\nRequired column availability:")

for col in required_columns:
    print(col, ":", col in df.columns)

# Stop if required columns are missing
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# --------------------------------------------------
# CLEAN DATA TYPES
# --------------------------------------------------

df["decimalLatitude"] = pd.to_numeric(
    df["decimalLatitude"], errors="coerce"
)

df["decimalLongitude"] = pd.to_numeric(
    df["decimalLongitude"], errors="coerce"
)

df["eventDate"] = pd.to_datetime(
    df["eventDate"],
    format="mixed",
    errors="coerce",
    utc=True
)

# --------------------------------------------------
# REMOVE DUPLICATES
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(subset=["id"])

print("\nDuplicate records removed:", before - len(df))

# --------------------------------------------------
# FILTER VALID COORDINATES
# --------------------------------------------------

before = len(df)

df = df[
    df["decimalLatitude"].between(-90, 90)
    & df["decimalLongitude"].between(-180, 180)
].copy()

print("Invalid coordinate rows removed:", before - len(df))

# --------------------------------------------------
# FILTER VALID DATES
# --------------------------------------------------

before = len(df)

df = df[
    df["eventDate"].notna()
    & df["eventDate"].between(
        pd.Timestamp("2015-01-01", tz="UTC"),
        pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    )
].copy()

print("Invalid/out-of-period dates removed:", before - len(df))

# --------------------------------------------------
# FILTER BASIC GEOGRAPHIC BOUNDS
# --------------------------------------------------

before = len(df)

valid_region = (
    (
        (df["query_region"] == "Arabian Sea")
        & df["decimalLongitude"].between(50, 78)
        & df["decimalLatitude"].between(0, 25)
    )
    |
    (
        (df["query_region"] == "Bay of Bengal")
        & df["decimalLongitude"].between(78, 100)
        & df["decimalLatitude"].between(0, 23)
    )
)

df = df[valid_region].copy()

print("Outside configured region bounds removed:", before - len(df))

# --------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------

output_file = OUTPUT_DIR / "obis_occurrences_clean.csv"

df.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print("Final rows:", len(df))
print("\nRecords by region and species:")

print(
    df.groupby(
        ["query_region", "common_name"]
    ).size().to_string()
)

print("\nMissing values in important fields:")

print(
    df[required_columns].isna().sum().to_string()
)

print("\nSaved:", output_file)
