import datetime

import json
from beemeteo.sources.era5 import ERA5

def test_era5():
    source = ERA5(json.load(open("config.json")))

    data = source.get_historical_grid_data(
        latitude=49.6,
        longitude=18.5,
        date_from=datetime.datetime(2013, 1, 2),
        date_to=datetime.datetime(2013, 2, 4)
    )

    source.collect_raster2(
        min_lat=40.5,
        max_lat=42.9,
        min_lon=0.1,
        max_lon=3.4,
        folder='data',
        ini=3
    )
