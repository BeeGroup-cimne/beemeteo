import datetime
import logging
from abc import abstractmethod
from datetime import timedelta

import pandas as pd
import pytz
from timezonefinder import TimezoneFinder

from beemeteo.utils import save_to_hbase, read_config, get_hbase_data_batch, _datetime_dt_to_ts_utc, _pandas_ts_to_dt, \
    _datetime_to_tz

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class Source:
    hbase_table_historical = None
    hbase_table_forecasting = None

    def __init__(self, config):
        self.config = read_config(config)

    @abstractmethod
    def _get_historical_data_source(self, latitude, longitude, gaps, local_tz):
        """
        Gets historical data from source
        :param latitude: station's latitude
        :param longitude: station's longitude
        :param gaps: the list of gaps to retrieve from the source
        :param local_tz: the local timezone of the station
        :return: all raw data for the given gaps
        """
        pass

    @abstractmethod
    def _collect_forecasting(self, latitude, longitude, now, tz_local):
        """
        Gets one day of forecast data from source
        :param latitude: station's latitude
        :param longitude: station's longitude
        :param gaps: the list of gaps to retrieve from the source
        :param local_tz: the local timezone of the station
        :return: all raw data for the given gaps
        """
        pass

    def _prepare_forecasting_input(self, latitude, longitude, date_from, date_to, tz_in_location, **kwargs):
        # set 3 dots in location
        latitude = format(latitude, '.3f')
        longitude = format(longitude, '.3f')

        # all dates must be in the timezone of the location
        date_from_local = _datetime_to_tz(date_from, tz_in_location)
        date_to_local = _datetime_to_tz(date_to, tz_in_location)
        g_ts_ini_utc = _datetime_dt_to_ts_utc(date_from_local)
        g_ts_end_utc = _datetime_dt_to_ts_utc(date_to_local)
        key_mapping = {"latitude": 0, "longitude": 1, "forecasting_timestamp": 2, "timestamp": 3}

        return latitude, longitude, g_ts_ini_utc, g_ts_end_utc, key_mapping

    def _parse_forecasting_output(self, data, tz_in_location, **kwargs):
        data.forecasting_timestamp = _pandas_ts_to_dt(data.forecasting_timestamp, tz_in_location)
        data.timestamp = data.timestamp.astype(int)
        data.timestamp = _pandas_ts_to_dt(data.timestamp, tz_in_location)
        return data

    @abstractmethod
    def _transform_data(self, latitude, longitude, date_from, date_to):
        """
        Transforms the data into the desired format
        """
        pass

    def collect_forecasting(self, latitude, longitude):
        """
        Collects a forecasting point from the data_source and stores it in HBASE.
        Many sources will need to call this function periodically to fill the forecasting database, for the sources
        that don't provide historical forecasting
        :param latitude: station's latitude
        :param longitude: station's longitude
        """
        # set 3 dots in location
        latitude = format(latitude, '.3f')
        longitude = format(longitude, '.3f')

        # all dates must be in the timezone of the location
        tz_find = TimezoneFinder()
        tz_in_location = pytz.timezone(tz_find.timezone_at(lat=float(latitude), lng=float(longitude)))
        now = pytz.UTC.localize(datetime.datetime.utcnow()).astimezone(tz_in_location)
        now = now.replace(minute=0, second=0, microsecond=0)
        forecasted_data = self._collect_forecasting(latitude, longitude, now, tz_in_location)
        forecasted_data = forecasted_data.query("timestamp >= {}".format(now.astimezone(pytz.UTC).timestamp())).\
            sort_values(by=["forecasting_timestamp", "timestamp"])
        save_to_hbase(forecasted_data.to_dict(orient="records"), self.hbase_table_forecasting,
                      self.config['hbase_weather_data'], [("info", "all")],
                      row_fields=["latitude", "longitude", "forecasting_timestamp", "timestamp"])

    def collect_raster(self,  min_lat, max_lat, min_lon, max_lon, day):
        """
        Collects a forecasting point from the data_source and stores it in HBASE.
        Many sources will need to call this function periodically to fill the forecasting database, for the sources
        that don't provide historical forecasting
        :param latitude: station's latitude
        :param longitude: station's longitude
        """
        forecasted_data = self._collect_raster(min_lat, max_lat, min_lon, max_lon, day)
        precision = 10

        forecasted_data['latitude'] = (round(forecasted_data['latitude']*precision)).astype(int)/precision
        forecasted_data['longitude'] = (round(forecasted_data['longitude']*precision)).astype(int)/precision

        dfs = {}

        for i in range(0, len(forecasted_data), 96):
            chunk = forecasted_data.iloc[i:i+96].reset_index(drop=True)
            lat = chunk.iloc[0]['latitude']
            lon = chunk.iloc[0]['longitude']
            if (lat, lon) not in dfs:
                dfs[(lat, lon)] = [chunk]
            else:
                dfs[(lat, lon)].append(chunk)

        cols_mean = ['totalPrecipitation', 'relativeHumidity', 'GHI', 'airTemperature','timestamp', 'latitude', 'longitude','forecasting_timestamp']
        cols_sum = ['u', 'v']

        for lat_lon in dfs.keys():
            df = pd.concat(dfs[lat_lon],axis=1)
            mean_df = df[cols_mean].T.groupby(level=0).mean().T
            sum_df = df[cols_sum].T.groupby(level=0).sum().T

            result_df = pd.concat([mean_df, sum_df], axis=1)
            result_df['forecasting_timestamp'] = result_df['forecasting_timestamp'].astype(int)
            result_df['timestamp'] = result_df['timestamp'].astype(int)

            save_to_hbase(result_df.to_dict(orient="records"), self.hbase_table_forecasting,
                          self.config['hbase_weather_data'], [("info", "all")],
                          row_fields=["latitude", "longitude", "timestamp", "forecasting_timestamp"])


    def get_forecasting_data(self, latitude, longitude, date_from, date_to, **kwargs):
        """
        Collects the forecasting data from the data_source and the forecasting for each available horizon
        :param latitude: station's latitude
        :param longitude: station's longitude
        :param date_from: start date
        :param date_to: end date
        """
        tz_find = TimezoneFinder()
        tz_in_location = pytz.timezone(tz_find.timezone_at(lat=float(latitude), lng=float(longitude)))
        args = self._prepare_forecasting_input(latitude, longitude, date_from, date_to, tz_in_location, **kwargs)

        data = self._get_from_hbase(*args, self.hbase_table_forecasting)
        return self._parse_forecasting_output(data, tz_in_location, **kwargs)

    def get_historical_data(self, latitude, longitude, date_from, date_to):
        """
        Gets historical data from source

        :param latitude: station's latitude
        :param longitude: station's longitude
        :param date_from: start date
        :param date_to: end date
        """
        # all dates must be in the timezone of the location
        tz_find = TimezoneFinder()
        tz_in_location = pytz.timezone(tz_find.timezone_at(lat=float(latitude), lng=float(longitude)))
        date_from_local = _datetime_to_tz(date_from, tz_in_location)
        date_to_local = _datetime_to_tz(date_to, tz_in_location)

        # get the current time in location
        now = _datetime_to_tz(pytz.UTC.localize(datetime.datetime.utcnow()), tz_in_location)
        if date_to_local > now:
            logger.info("Warning: requested date {} is newer than current date {}".format(date_to_local, now))

        g_ts_ini_utc = _datetime_dt_to_ts_utc(date_from_local)
        g_ts_end_utc = _datetime_dt_to_ts_utc(date_to_local)

        # set 3 dots in location
        latitude = format(latitude, '.3f')
        longitude = format(longitude, '.3f')

        # get data from hbase, the date end must be fully included
        key_mapping = {"latitude": 0, "longitude": 1, "ts": 2}
        data = self._get_from_hbase(latitude, longitude, g_ts_ini_utc, g_ts_end_utc + 86400,
                                    key_mapping, self.hbase_table_historical)
        if not data.empty:
            data['ts'] = data['ts'].astype(int)
        key_cols = ["latitude", "longitude", "ts"]
        data = data.set_index(key_cols)[
            sorted(data.columns[~data.columns.isin(key_cols)])
        ].reset_index() if not data.empty else data
        # check if the data contains all hours, if not, get the missing days
        gaps = []
        if data.empty:
            gaps = [(date_from_local, date_to_local)]
        else:
            data['ts2_py'] = _pandas_ts_to_dt(data['ts'], tz_in_location)
            if date_from_local < data['ts2_py'].min():
                gaps.append((date_from_local, data['ts2_py'].min().to_pydatetime()))
            if min(date_to_local, now) - timedelta(hours=1) > data['ts2_py'].max():
                gaps.append((data['ts2_py'].max().to_pydatetime(), min(date_to_local, now)))
            deltas = data['ts2_py'].diff()[1:]
            gaps_tmp = deltas[deltas > timedelta(hours=1)]
            for i, _ in gaps_tmp.iteritems():
                gap_start = data['ts2_py'][i - 1]
                gap_end = data['ts2_py'][i]
                gaps.append((gap_start, gap_end))
            data = data.drop('ts2_py', axis=1)
        if gaps:
            missing_data = self._get_historical_data_source(latitude, longitude, gaps, tz_in_location)
            if not missing_data.empty:
                missing_data = missing_data.query("ts <= {}".format(_datetime_dt_to_ts_utc(now))).sort_values(by=["ts"])
                save_to_hbase(missing_data.to_dict(orient="records"), self.hbase_table_historical,
                              self.config['hbase_weather_data'], [("info", "all")],
                              row_fields=["latitude", "longitude", "ts"])
            data = pd.concat([data, missing_data]) if not data.empty else missing_data
        if not data.empty:
            data = data.query("ts >= {} and ts <= {}".format(g_ts_ini_utc,
                                                             min(_datetime_dt_to_ts_utc(now),
                                                                 g_ts_end_utc + 82800))).sort_values(by=["ts"])
            data.drop_duplicates(subset=['latitude', 'longitude', 'ts'], inplace=True)
            data.ts = _pandas_ts_to_dt(data.ts, tz_in_location)
        return data

    def _get_from_hbase(self, latitude, longitude, ts_ini, ts_end, key_mapping, hbase_table):
        """
        Gets all raw data from HBase for a given day

        :param latitude: station's latitude
        :param longitude: station's longitude
        :param ts_ini: a given timestamp to start getting data
        :param ts_end: a given timestamp to stop getting data
        :return:
        """
        measures = []
        try:
            for data_tmp in get_hbase_data_batch(self.config['hbase_weather_data'], hbase_table, columns=["info"],
                                                 row_start="%s~%s~%d" % (latitude, longitude, ts_ini),
                                                 row_stop="%s~%s~%d" % (latitude, longitude, ts_end)):
                for row_key, data in data_tmp:
                    new_data = data.copy()
                    for key, n_key in \
                            zip(data.keys(), [key.decode("utf-8").replace("info:", "") for key in data.keys()]):
                        new_data[n_key] = new_data.pop(key).decode("utf-8")
                    for k, v in key_mapping.items():
                        new_data[k] = row_key.decode("UTF-8").split("~")[v]
                    measures.append(new_data)
            return pd.DataFrame(measures)
        except Exception as e:
            return pd.DataFrame({})
