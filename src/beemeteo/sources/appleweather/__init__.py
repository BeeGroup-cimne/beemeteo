import requests
import sys
import json
import datetime
import pytz
import pandas as pd

from beemeteo.utils import _local_to_UTC, _UTC_to_local, _datetime_to_api_format, _api_format_to_datetime 
from beemeteo.sources import Source, logger


class AppleWeather(Source):
    hbase_table_historical = "darksky_historical"
    hbase_table_forecasting = "darksky_forecasting"

    def __init__(self, config):
        super(AppleWeather, self).__init__(config)
        self.api_key = self.config["apple_weather"]["token"]

    def _collect_forecasting(self, latitude, longitude, now, local_tz):
        later = now + datetime.timedelta(days=3)
        df = self._request_server(latitude, longitude, "forecastHourly", now, later, local_tz)
        df.rename(columns={'forecastStart': 'ts'}, inplace=True)
        df = self._to_darksky_format(df, latitude, longitude)
        df["forecasting_timestamp"] = int(now.astimezone(pytz.UTC).timestamp())
        df.rename(columns={'ts': 'timestamp'}, inplace=True)
        df['timestamp']= df['timestamp'].astype(int).div(10**9).astype(int)
        return df 

    def _get_historical_data_source(self, latitude, longitude, gaps, local_tz):
        # Apple Weather will return hourly data in UTC time.
        # https://developer.apple.com/forums/thread/722722
        df = pd.DataFrame()                                                                                            
        for ts_ini, ts_end in gaps:
            data_period = self._request_server(latitude, longitude, "forecastHourly", ts_ini, ts_end, local_tz)
            df = pd.concat([df, data_period])
        df.rename(columns={'forecastStart': 'ts'}, inplace=True)
        df = self._to_darksky_format(df, latitude, longitude)
        df['ts']= df['ts'].astype(int).div(10**9).astype(int)
        return df 

    def _request_server(self, lat, long, service, day_from, day_to, local_tz):
        weather_url = 'https://weatherkit.apple.com/api/v1/weather/en_US/'
        day_from = _local_to_UTC(day_from, local_tz)
        day_to = _local_to_UTC(day_to, local_tz)
        url = weather_url + str(lat) + '/' + str(long)
        headers = {'Authorization': 'Bearer {}'.format(self.api_key)}
    
        payload = {
                "dataSets": service,
                "hourlyStart": day_from,
                "hourlyEnd": day_to,
                "timezone": local_tz,
                }

        data_list = []
        if service == "currentWeather":
            day_aux = day_to
        else: 
            day_aux = day_from
        # To include up to 23:59 we must set the  end date to the following day, 00:00
        day_to += datetime.timedelta(days=1)

        while day_to > day_aux:
            period_days = (day_to - day_aux).days
            payload["hourlyStart"] = _datetime_to_api_format(day_aux)
            # requests of maximum 10 days
            day_aux += datetime.timedelta(days=min(period_days, 10))
            payload["hourlyEnd"] = _datetime_to_api_format(day_aux) 
            response = requests.get(url, headers=headers, params=payload)
            if response.status_code != 200:
                print(f"The request from {payload['hourlyStart']} to"
                      f" {payload['hourlyEnd']} failed with: {response.reason}", file=sys.stderr)
                continue
            if not json.loads(response.text):
                print(f"The request from {payload['hourlyStart']} to"
                      f" {payload['hourlyEnd']} is empty: {response.reason}", file=sys.stderr)
                continue
            data_list += self._parse_request(response.text)

        for d in data_list:
            d.update((k, _api_format_to_datetime(v)) for k, v in d.items() if (k == 'forecastStart' or k == 'asOf'))
            d.update((k, _UTC_to_local(v, local_tz)) for k, v in d.items() if (k == 'forecastStart' or k == 'asOf'))

        return pd.DataFrame.from_records(data_list)

    @staticmethod
    def _parse_request(response):
        """
        Parse the request output into a pandas DataFrame.
        :param response: api response in txt format
        :return: python list
        """
    
        data = json.loads(response)
        service = next(iter(data))
        data = data[service]
        del data['name'], data['metadata']
        return(
                data['hours']
                if(service != 'currentWeather')
                else [data]
                )

    @staticmethod
    def _to_darksky_format(data, latitude, longitude):
        """
        Adapt retrieved data to DarkSky frame
        :data: Pandas dataframe after prasing
        :latitude: Latitude of retrieved location (not string)
        :longitude: Longitude of retrieved location (not string)
        """
        apple_dark_map = {
            "temperatureApparent": "apparentTemperature",
            "temperatureDewPoint": "dewPoint",
            "conditionCode": "icon",
            "precipitationAmount": "precipAccumulation",
            "precipitationIntensity": "precipIntensity",
            "precipitationChance": "precipProbability",
            "precipitationType": "precipType",
            "windDirection": "windBearing",
            "dailyStart": "time", "hourlyStart": "time", "forecastStart": "time", "startTime": "time",
            "sunset": "sunsetTime", "sunsetAstronomical": "sunsetTime", "sunsetCivil": "sunsetTime",
            "sunsetNautical": "sunsetTime",
            "sunrise": "sunriseTime", "sunriseAstronomical": "sunriseTime", "sunriseCivil": "sunriseTime",
            "sunriseNautical": "sunriseTime",

        }
        if data.empty:
            return data
        data = data.sort_values(by=["ts"])
        data['latitude'] = latitude
        data['longitude'] = longitude
        data.drop_duplicates(subset=['latitude', 'longitude', 'ts'], inplace=True)
        key_cols = ["latitude", "longitude", "ts"]
        data = data.set_index(key_cols)[
            sorted(data.columns[~data.columns.isin(key_cols)])
        ].reset_index() if not data.empty else data
        data = data.rename(apple_dark_map, axis=1)
        return data
