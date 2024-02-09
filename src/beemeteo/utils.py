import datetime
import json
import logging
import time
import uuid

import happybase
import pandas as pd
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pandas_to_tz(dt, timezone):
    return (
        dt.dt.tz_convert(timezone)
        if dt.dt.tz is not None
        else
        dt.dt.tz_localize(timezone)
    )


def _pandas_dt_to_ts_utc(dt):
    if dt.dt.tz is None:
        raise Exception("The datetime must have a timezone")
    ts_init = pd.Timestamp("1970-01-01").tz_localize(pytz.UTC)
    return ((_pandas_to_tz(dt, pytz.UTC) - ts_init) / pd.Timedelta("1s")).astype(int)


def _pandas_ts_to_dt(ts, timezone):
    ts = pd.to_datetime(ts, unit="s")
    return ts.dt.tz_localize(pytz.UTC).dt.tz_convert(timezone)


def _datetime_to_tz(dt, timezone):
    return (
        dt.astimezone(timezone)
        if dt.tzinfo is not None
        else timezone.localize(dt)
    )


def _datetime_dt_to_ts_utc(dt):
    if dt.tzinfo is None:
        raise Exception("The datetime must have a timezone")
    return int((_datetime_to_tz(dt, pytz.UTC) - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)) /
               datetime.timedelta(seconds=1))

def _local_to_UTC(date, local_tz):
    return pytz.utc.normalize(date)

def _UTC_to_local(date, local_tz):
    return pytz.UTC.localize(date).astimezone(local_tz)

def _datetime_to_api_format(date):
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")

def _api_format_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")


def read_config(config):
    return config


def __get_h_table__(hbase, table_name, cf=None):
    try:
        if not cf:
            cf = {"cf": {}}
        hbase.create_table(table_name, cf)
    except Exception as e:
        if str(e.__class__) == "<class 'Hbase_thrift.AlreadyExists'>":
            pass
        else:
            print(e)
    return hbase.table(table_name)


def save_to_hbase(documents, h_table_name, hbase_connection, cf_mapping, row_fields=None, batch_size=1000):
    hbase = happybase.Connection(**hbase_connection)
    table = __get_h_table__(hbase, h_table_name, {cf: {} for cf, _ in cf_mapping})
    h_batch = table.batch(batch_size=batch_size)
    row_auto = 0
    uid = uuid.uuid4()
    for d in documents:
        if not row_fields:
            row = f"{uid}~{row_auto}"
            row_auto += 1
        else:
            row = "~".join([str(d.pop(f)) if f in d else "" for f in row_fields])
        values = {}
        for cf, fields in cf_mapping:
            if fields == "all":
                for c, v in d.items():
                    values["{cf}:{c}".format(cf=cf, c=c)] = str(v)
            else:
                for c in fields:
                    if c in d:
                        values["{cf}:{c}".format(cf=cf, c=c)] = str(d[c])
        h_batch.put(str(row), values)
    h_batch.send()


def get_hbase_data_batch(hbase_conf, hbase_table, row_start=None, row_stop=None, row_prefix=None, columns=None,
                         _filter=None, timestamp=None, include_timestamp=False, batch_size=100000,
                         scan_batching=None, limit=None, sorted_columns=False, reverse=False):

    if row_prefix:
        row_start = row_prefix
        row_stop = row_prefix[:-1]+chr(row_prefix[-1]+1).encode("utf-8")

    if limit:
        if limit > batch_size:
            current_limit = batch_size
        else:
            current_limit = limit
    else:
        current_limit = batch_size
    current_register = 0
    while True:
        hbase = happybase.Connection(**hbase_conf)
        table = hbase.table(hbase_table)
        data = list(table.scan(row_start=row_start, row_stop=row_stop, columns=columns, filter=_filter,
                               timestamp=timestamp, include_timestamp=include_timestamp, batch_size=batch_size,
                               scan_batching=scan_batching, limit=current_limit, sorted_columns=sorted_columns,
                               reverse=reverse))
        if not data:
            break
        last_record = data[-1][0]
        current_register += len(data)
        yield data

        if limit:
            if current_register >= limit:
                break
            else:
                current_limit = min(batch_size, limit - current_register)
        row_start = last_record[:-1] + chr(last_record[-1] + 1).encode("utf-8")
    yield []
