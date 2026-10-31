
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

regions = ["arabian_sea", "bay_of_bengal"]

for region in regions:

    path = (
        ROOT / "data" / "processed" / "gfw"
        / region / "gfw_2019_2025.csv"
    )

    df = pd.read_csv(path)

    print("\n==========", region.upper(), "==========")

    print("Negative fishing hours:", (df["hours"] < 0).sum())
    print("Zero fishing hours:", (df["hours"] == 0).sum())
    print("Negative vessel IDs:", (df["vesselIDs"] < 0).sum())

    print("Missing flags:", df["flag"].isna().sum())

    print("\nTop 10 flags:")
    print(df["flag"].value_counts().head(10))

    print("\nMonthly coverage:")
    print("Months present:", df["date"].nunique())
    print("Expected months: 84")

    print("\nRecords with missing flags:")
    print(df[df["flag"].isna()].head())

    print("\nHours statistics:")
    print(df["hours"].describe())
