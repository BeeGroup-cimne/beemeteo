# beemeteo

[![CI](https://github.com/beedata-analytics/beemeteo/actions/workflows/main.yml/badge.svg)](https://github.com/beedata-analytics/beemeteo/actions/workflows/main.yml)
[![Python version](https://img.shields.io/badge/python-2.7-blue)](https://img.shields.io/badge/python-2.7-blue)
[![Python version](https://img.shields.io/badge/python-3.8-blue)](https://img.shields.io/badge/python-3.8-blue)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/beedata-analytics/beedatadis/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/beedata-analytics/beedatadis/blob/master/.pre-commit-config.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)


A python package and command line tool to download weather data from different sources.

## Sources Available

- [CAMS Radiation Service](http://www.soda-pro.com/web-services/radiation/cams-radiation-service/info) (CAMS)
  - Provides Europe and Africa historical solar radiation data ![OK](https://img.shields.io/badge/-OK-green)
- [Dark Sky API](https://darksky.net/dev) (DarkSky)
  - Provides worldwide historical meteorological data ![OK](https://img.shields.io/badge/-deprecated-red)
  - Provides worldwide prediction meteorological data ![NO](https://img.shields.io/badge/-deprecated-red)
- [Apple Weather kit"](https://developer.apple.com/weatherkit/) (AppleWeatherKit)
  - Provides worldwide historical meteorological data ![OK](https://img.shields.io/badge/-OK-green)
  - Provides worldwide prediction meteorological data ![OK](https://img.shields.io/badge/-OK-green)
- [MeteoGalicia](http://mandeo.meteogalicia.es) (MeteoGalicia)
  - Provides Europe historical solar radiation data ![OK](https://img.shields.io/badge/-OK-green)
  - Provides Europe prediction solar radiation data ![NO](https://img.shields.io/badge/-OK-green)
## Installation

To install the package, use pip and the Git repository at the newest version

```bash
pip install git+https://github.com/BeeGroup-cimne/beemeteo@v0.3.1
```

## Config file
To use the tool, you will need a json configuration file as the following
```json
{
  "cams":{
    "registered_emails": ["list of registered e-mails"]
  },
  "dark_sky": {
    "api_key": "api key to access darksky(deprecated)"
  },
  "hbase_weather_data": {
    "host": "hbase host",
    "port": 9090,
    "table_prefix": "hbase namespace",
    "table_prefix_separator": ":"
  }
}
```
## Usage
The application can be used as a command line or imported in a python package.

### Timezones

When downloading data from a location, the application will detect the timezone of the location and download and return the data in the location's timezone.

If the passed datetime don't contain any timezone (they are naive). The location's timezone will be used as their timezone

If the passed datetime contains a timezone (not naive). The date will be translated to the location's timezone to perform the downloading.

### command line

To use the command line, the python package must be installed in the python environment of your computer. 

The command line has 3 different applications get data: historical data and forecasting data, and collect_forecasting

```console
python3 -m beemeteo --help

usage: __main__.py [-h] {historical,forecasting,collect_forecasting} ...

positional arguments:
  {historical,forecasting,collect_forecasting}
                        The type of data to collect
    historical          get historical data
    forecasting         get forecasting data
    collect_forecasting
                        collect forecasting data for now

optional arguments:
  -h, --help            show this help message and exit

```

Both the execution of the command line program for get forecasting or historical data, will generate a CSV file with the specified name in the "file" parameter
```console
$ python3 -m beemeteo historical/forecasting --help

usage: __main__.py [-h] -s SOURCE -c CONFIG -lat LATITUDE -lon LONGITUDE -d1 DATE_FROM -d2 DATE_TO -f FILE

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        The source to get data from one of [AppleWeather, CAMS, MeteoGalicia]
  -c CONFIG, --config CONFIG
                        The configuration file path
  -lat LATITUDE, --latitude LATITUDE
                        The latitude
  -lon LONGITUDE, --longitude LONGITUDE
                        The longitude
  -d1 DATE_FROM, --date_from DATE_FROM
                        The date to get weather from
  -d2 DATE_TO, --date_to DATE_TO
                        The date to get weather to (included)
  -f FILE, --file FILE  The file to store the data

Example: python3 -m beemeteo \
  historical -s AppleWeather \
  -c config.json -lat 41.29 -lon 2.19 -d1 2021-09-01 
  -d2 2021-09-05 --f data_file.csv
```
The collect forecasting will not return any information, but is used to populate the forecasting data with the current hour.

```console
python3 -m beemeteo collect_forecasting --help

usage: __main__.py collect_forecasting [-h] -s {AppleWeather,CAMS,MeteoGalicia} -c
                                       CONFIG -lat LATITUDE -lon LONGITUDE

optional arguments:
  -h, --help            show this help message and exit
  -s {AppleWeather,CAMS,MeteoGalicia}, --source {AppleWeather,CAMS,MeteoGalicia}
                        The source to get data from
  -c CONFIG, --config CONFIG
                        The configuration file path
  -lat LATITUDE, --latitude LATITUDE
                        The latitude
  -lon LONGITUDE, --longitude LONGITUDE
                        The longitude
Example: python3 -m beemeteo \
collect_forecasting \
-s AppleWeather \
-c config.json \
-lat 41.29 \
-lon 2.19 

```


### python package
To use the package in python, you have to import the different data_sources, and use the functions to use them

#### cams

```python
import datetime
import os
import pytz
from beemeteo.sources.cams import CAMS

source = CAMS("config.json")
source.get_historical_data(
    41.29,
    2.19,
    datetime.datetime(2021, 9, 1),
    datetime.datetime(2021, 9, 5),
)
```

#### darksky

```python
import datetime
import os

import pytz

from beemeteo.sources.appleweather import AppleWeather

source = AppleWeather("config.json")
source.get_historical_data(
    41.29,
    2.19,
    datetime.datetime(2021, 9, 1),
    datetime.datetime(2021, 9, 5),
)
source.get_forecasting_data(
    41.29,
    2.19,
    datetime.datetime(2021, 9, 1),
    datetime.datetime(2021, 9, 5),
)
source.collect_forecasting(
    41.29,
    2.19,
)
```

#### meteogalicia

```python
import datetime
import pytz
from beemeteo.sources.meteogalicia import MeteoGalicia
source = MeteoGalicia("config.json")
source.get_historical_data(
    41.29,
    2.19,
    datetime.datetime(2021, 9, 1),
    datetime.datetime(2021, 9, 5),
)
source.get_forecasting_data(
    41.29,
    2.19,
    datetime.datetime(2021, 9, 1),
    datetime.datetime(2021, 9, 5),
)
source.collect_forecasting(
    41.29,
    2.19,
)
```

#### ERA5

```python
import datetime

import json
from beemeteo.sources.era5 import ERA5

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
```

## Authors
- Jose Manuel Broto - jmbroto@cimne.upc.edu
- Eloi Gabaldón - egabaldon@cimne.upc.edu
- Arian Huapaya - ahuapaya@cimne.upc.edu
  
Copyright (c) 2024 Jose Manuel Broto, Eloi Gabaldón, Arian Huapaya
