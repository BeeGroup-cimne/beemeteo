import cdsapi
import os
import pandas as pd
import numpy as np
import sys
import cfgrib
import pytz
from tqdm import tqdm
import functools as ft
import random


# Define the variables that should be obtained from the ERA5-Land dataset.
# Check the units and description: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview
# Check the long and short names used in CDS API: https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation (table 2)
def ERA5Land_variables():
    return [
        {
            'CDSLongName': '10m_u_component_of_wind',
            'CDSShortName': 'u10',
            'name': 'windSpeedEast'
        },
        {
            'CDSLongName': '10m_v_component_of_wind',
            'CDSShortName': 'v10',
            'name': 'windSpeedNorth'
        },
        {
            'CDSLongName': '2m_dewpoint_temperature',
            'CDSShortName': 'd2m',
            'name': 'dewAirTemperature'
        },
        {
            'CDSLongName': '2m_temperature',
            'CDSShortName': 't2m',
            'name': 'airTemperature'
        },
        {
            'CDSLongName': 'leaf_area_index_high_vegetation',
            'CDSShortName': 'lai_hv',
            'name': 'highVegetationRatio'
        },
        {
            'CDSLongName': 'leaf_area_index_low_vegetation',
            'CDSShortName': 'lai_lv',
            'name': 'lowVegetationRatio'
        },
        {
            'CDSLongName': 'total_precipitation',
            'CDSShortName': 'tp',
            'name': 'totalPrecipitation'
        },
        {
            'CDSLongName': 'surface_solar_radiation_downwards',
            'CDSShortName': 'ssrd',
            'name': 'GHI'
        },
        {
            'CDSLongName': 'forecast_albedo',
            'CDSShortName': 'fal',
            'name': 'albedo'
        },
        {
            'CDSLongName': 'soil_temperature_level_4',
            'CDSShortName': 'stl4',
            'name': 'soilTemperature'
        },
        {
            'CDSLongName': 'volumetric_soil_water_layer_4',
            'CDSShortName': 'swvl4',
            'name': 'soilWaterRatio'
        }]


# Function to gather weather data from ERA5-Land
def get_hourly_historical_weather_from_ERA5Land(data_dir, ym_range, lat_range=[90, -90], lon_range=[-180, 180]):
    # Connect to the Copernicus Climate Date Store
    try:
        c = cdsapi.Client()  # You'll need a CDS credentials file in ~/.cdsapirc
    except:
        raise print("Obtain credentials to Climate Data Store and set them in ~/.cdsapirc file. " \
                    "Please, follow instructions in: " \
                    "https://cds.climate.copernicus.eu/api-how-to#install-the-cds-api-key", file=sys.stderr)

    # Gather the GRIB file for each year-month.
    ym = min(ym_range)
    os.makedirs(data_dir, exist_ok=True)
    while ym <= max(ym_range):
        year = int(str(ym)[0:4])
        month = int(str(ym)[4:6])
        filename = f'{data_dir}/{year}{month:02}_{max(lat_range)}_{min(lon_range)}_{min(lat_range)}_{max(lon_range)}.grib'
        sys.stderr.write(f'--> Obtaining {year}{month:02} in bounding box: {max(lat_range)},{min(lon_range)} ' \
                         f'and {min(lat_range)},{max(lon_range)}\n')
        if not os.path.exists(filename):
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [x['CDSLongName'] for x in ERA5Land_variables()],
                    'year': f'{year}',
                    'month': f'{month:02}',
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'area': [  # Bounding box
                        max(lat_range), min(lon_range),  # Upper left point (Lat-lon)
                        min(lat_range), max(lon_range)  # Lower right point (Lat-lon)
                    ],
                    'format': 'grib',
                },
                filename)
        sys.stderr.write('Done!\n')
        month += 1
        if month > 12:
            year += 1
            month = 1
        ym = int(f'{year}{month:02}')

    return sys.stderr.write('Success! All data has been downloaded.\n')


def cleaning_pipe(df, lat, lon):
    df = df.dropna()
    df = df.reset_index()
    if lat is not None or lon is not None:
        df = df[(df["latitude"].round(1) == np.round(lat, 1)) & (df.longitude.round(1) == np.round(lon, 1))]
    elif lon is not None:
        df = df[df.longitude.round(1) == np.round(lon, 1)]
    elif lat is not None:
        df = df[df.latitude.round(1) == np.round(lat, 1)]
    df = df.drop(['time', 'step', 'number', 'surface'], axis=1, errors='ignore')
    df = df.rename(dict(zip(['valid_time'] + [x['CDSShortName'] for x in ERA5Land_variables()],
                            ['time'] + [x['name'] for x in ERA5Land_variables()])), axis=1)
    return df


def transformation_pipe(df):
    df["time"] = df["time"].dt.tz_localize(pytz.UTC)
    df["windSpeed"] = np.sqrt(df["windSpeedEast"] ** 2 + df["windSpeedNorth"] ** 2)
    df["windDirection"] = (np.arctan2(df["windSpeedNorth"], df["windSpeedEast"]) * 180 / np.pi) + 180
    df["soilTemperature"] = df["soilTemperature"] - 273.15
    df["dewAirTemperature"] = df["dewAirTemperature"] - 273.15
    df["airTemperature"] = df["airTemperature"] - 273.15
    df["relativeHumidity"] = (6.112 * np.exp(
        (17.67 * df["dewAirTemperature"] / (df["dewAirTemperature"] + 243.5)))) * 100 / \
                             (6.112 * np.exp((17.67 * df["airTemperature"] / (df["airTemperature"] + 243.5))))
    df["GHI"] = (np.array(df["GHI"]) - shift(df["GHI"], 1))
    df["GHI"] = np.where(df["GHI"] > 0, df["GHI"] / 3600, 0)
    df["totalPrecipitation"] = (np.array(df["totalPrecipitation"]) - shift(df["totalPrecipitation"], 1))
    df["totalPrecipitation"] = np.where(df["totalPrecipitation"] > 0, df["totalPrecipitation"] / 3600, np.nan)

    return df


def query_from_grib(data_dir, lat=None, lon=None, grib_contains=None):
    list_grib_files = sorted([f'{data_dir}/{fn}' for fn in os.listdir(data_dir) if
                              fn.endswith(".grib") and
                              (lat is None or float(os.path.splitext(fn)[0].split('_')[1]) >= lat >= float(
                                  os.path.splitext(fn)[0].split('_')[3])) and
                              (lon is None or float(os.path.splitext(fn)[0].split('_')[2]) <= lon <= float(
                                  os.path.splitext(fn)[0].split('_')[4])) and
                              (grib_contains is None or grib_contains in fn)
                              ])

    df = None
    l = len(list_grib_files)

    for i in tqdm(range(l), desc="Loading..."):

        fn = list_grib_files[i]

        hash = random.getrandbits(128)

        dg = cfgrib.open_datasets(fn, backend_kwargs={'indexpath': f'{fn}.{hash}.idx'})
        os.remove(f'{fn}.{hash}.idx')
        df_ = [dg[j].to_dataframe() for j in range(len(dg))]

        df_ = [dfi.pipe(cleaning_pipe, lat, lon) for dfi in df_]
        df_ = ft.reduce(lambda x, y: pd.merge(x, y, how='left', on=['latitude', 'longitude', 'time']), df_)
        df_ = df_.pipe(transformation_pipe)

        if df is not None:
            df = pd.concat([df, df_])
        else:
            df = df_

    return df


# Shift function
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr

    return result
