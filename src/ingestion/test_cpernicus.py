
import copernicusmarine
from pathlib import Path

output_dir = Path("data/raw/copernicus")
output_dir.mkdir(parents=True, exist_ok=True)

copernicusmarine.subset(
    dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",

    variables=[
        "thetao",
        "so",
        "uo",
        "vo"
    ],

    minimum_longitude=50,
    maximum_longitude=78,

    minimum_latitude=0,
    maximum_latitude=25,

    start_datetime="2015-01-01T00:00:00",
    end_datetime="2015-01-31T23:59:59",

    minimum_depth=0,
    maximum_depth=5,

    output_filename=str(
        output_dir / "arabian_sea_january_2015.nc"
    )
)

print("Copernicus test download completed!")

