import datetime

import pandas as pd
import json
from beemeteo.sources.meteogalicia import MeteoGalicia


def test_meteogalicia_historic():
    source = MeteoGalicia(json.load(open("config.json")))
    data = source.get_historical_data(
        latitude=41.29,
        longitude=2.19,
        date_from=datetime.datetime(2023, 9, 16),
        date_to=datetime.datetime(2023, 9, 18),
    )
    expected = pd.read_csv("tests/b2back/meteogalicia.csv")
    print(data)
    assert data.equals(expected)

def test_meteogalicia_forecast():
    source = MeteoGalicia(json.load(open("config.json")))
    latitude=41.29
    longitude=2.19
    expected = pd.read_csv("tests/b2back/meteogalicia.csv")
    source.collect_forecasting(latitude, longitude)
    forecast = source.get_forecasting_data(latitude, longitude,
                                           date_from = datetime.datetime.now()-datetime.timedelta(hours=5), 
                                           date_to = datetime.datetime.now())
    print(forecast)
    assert forecast.equals(expected)