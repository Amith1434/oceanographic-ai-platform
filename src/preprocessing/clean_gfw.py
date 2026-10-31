
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

regions = ["arabian_sea", "bay_of_bengal"]

for region in regions:

    input_file = (
        ROOT / "data" / "processed" / "gfw"
        / region / "gfw_2019_2025.csv"
    )

    output_file = (
        ROOT / "data" / "processed" / "gfw"
        / region / "gfw_clean.csv"
    )

    df = pd.read_csv(input_file)

    # Preserve the original CSV.
    # Replace missing flag values only in the cleaned copy.
    df["flag"] = df["flag"].fillna("Unknown")

    # Ensure correct data types
    df["hours"] = pd.to_numeric(df["hours"], errors="raise")
    df["vesselIDs"] = pd.to_numeric(
        df["vesselIDs"], errors="raise"
    )

    df["lat"] = pd.to_numeric(df["lat"], errors="raise")
    df["lon"] = pd.to_numeric(df["lon"], errors="raise")

    # Save cleaned dataset
    df.to_csv(output_file, index=False)

    print("\nRegion:", region)
    print("Records:", len(df))
    print("Missing values:", df.isnull().sum().sum())
    print("Unknown flags:", (df["flag"] == "Unknown").sum())
    print("Saved:", output_file)
