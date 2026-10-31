
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

regions = ["arabian_sea", "bay_of_bengal"]

for region in regions:

    path = (
        ROOT / "data" / "processed" / "gfw"
        / region / "gfw_2019_2025.csv"
    )

    print("\n" + "=" * 50)
    print("REGION:", region)
    print("=" * 50)

    if not path.exists():
        print("ERROR: CSV not found:", path)
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("Date range:", df["date"].min(), "to", df["date"].max())

    print("Latitude:", df["lat"].min(), "to", df["lat"].max())
    print("Longitude:", df["lon"].min(), "to", df["lon"].max())

    print("Unique flags:", df["flag"].nunique())
    print("Unique grid cells:", df[["lat", "lon"]].drop_duplicates().shape[0])

    print("Total fishing hours:", df["hours"].sum())

    print("Monthly record counts:")
    print(df.groupby("date").size().to_string())
