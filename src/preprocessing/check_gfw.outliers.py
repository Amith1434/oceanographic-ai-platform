
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

    print("\nTop 15 records by fishing hours:")

    top = df.nlargest(15, "hours")

    print(
        top[
            ["date", "flag", "hours", "lat",
             "lon", "vesselIDs"]
        ].to_string(index=False)
    )

    print("\nFishing hours percentiles:")

    print(
        df["hours"].quantile(
            [0.50, 0.90, 0.95, 0.99, 0.999, 1.0]
        )
    )
