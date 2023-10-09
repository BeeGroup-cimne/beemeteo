import datetime
import pandas as pd
import json

from beemeteo.sources.appleweather import AppleWeather


def test_apple_weather_historic():
    source = AppleWeather(json.load(open("config.json")))
    latitude = 41.540
    longitude = 2.454
    data = source.get_historical_data(
        latitude=latitude,
        longitude=longitude,
        date_from=datetime.datetime(2023, 9, 16),
        date_to=datetime.datetime(2023, 9, 18),
    )
    expected = pd.read_csv("tests/b2back/darksky.csv")
    columns_to_compare = [
        "latitude",
        "longitude",
        "ts",
        "windBearing",
        "windGust",
        "windSpeed",
    ]
    print(data)
    assert data.equals(expected)

def test_apple_weather_forecast():
    source = AppleWeather(json.load(open("config.json")))
    latitude = 41.540
    longitude = 2.454
    source.collect_forecasting(latitude, longitude)
    forecast = source.get_forecasting_data(latitude, longitude,
                                           date_from = datetime.datetime.now()-datetime.timedelta(days=5), 
                                           date_to = datetime.datetime.now())

    print(forecast)
    assert forecast.equals(expected)
