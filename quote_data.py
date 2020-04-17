import symbols
import json
import argparse
import os
import time
import logging
import pandas as pd

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
from pathlib import Path

log = logging.getLogger('kwantifiable')

def update_all_symbols(time_window):
    for symbol in symbols.symbols_list:
        log.info("Updating {} {}".format(symbol, time_window))
        update_symbol(symbol, time_window)
        time.sleep(16)  # 4 requests per min    

def update_symbol(symbol, time_window):
    # TODO update, not clear
    save_symbol(symbol, time_window)

def clear_and_get_full_data(time_window):
    path = Path("quotes/")
    if os.path.isdir(path): 
        del_dir(path)
    path.mkdir(parents=True, exist_ok=True)
    for symbol in symbols.symbols_list:
        log.info("Downloading {} {}".format(symbol, time_window))
        save_symbol(symbol, time_window)
        time.sleep(15)  # 4 requests per min

def clear(path_name):
    path = Path(path_name)
    if os.path.isdir(path): 
        del_dir(path)

def del_dir(target):
    target = Path(target).expanduser()
    assert target.is_dir()
    for p in sorted(target.glob('**/*'), reverse=True):
        if not p.exists():
            continue
        p.chmod(0o666)
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()
        target.rmdir()


def save_symbol(symbol, time_window):
    credentials = json.load(open('credentials.json', 'r'))
    api_key = credentials['av_key']
    symbol = symbol.upper()
    ts = TimeSeries(key=api_key, output_format='pandas')
    if time_window == 'intraday':
        data, meta_data = ts.get_intraday(
            symbol=symbol, interval='1min', outputsize='full')
    elif time_window == 'daily':
        data, meta_data = ts.get_daily(symbol, outputsize='full')
    elif time_window == 'daily_adj':
        data, meta_data = ts.get_daily_adjusted(symbol, outputsize='full')

    pprint(data.head(10))

    data.to_csv(f'./quotes/{symbol}_{time_window}.csv')

def get_symbol(symbol, time_window):
    data = pd.read_csv(f'./quotes/{symbol}_{time_window}.csv')
    # drop last row to remove first day trading volume skew    
    data = data[:-1]

    # drop the original close price (as we have adj close), dividend and split coef
    data = data.drop(['date', '4. close', '7. dividend amount','8. split coefficient'], axis=1)

    return data.values

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('action', type=str, help="Action to perform", choices=['save', 'update', 'clear', 'updateall'])
    parser.add_argument('--symbol', type=str, help="the stock symbol you want to download", default=None)
    parser.add_argument('--time_window', type=str, choices=[
                        'intraday', 'daily', 'daily_adj'], default=None, help="the time period you want to download the stock history for")

    namespace = parser.parse_args()
    inputs = vars(namespace)
    print(inputs)

    if inputs['action'] == 'clear':
        clear()
    elif inputs['action'] == 'save':
        if inputs['symbol'] == None or inputs['time_window'] == None:
            print("must provide symbol and time window for update or save")
            exit()
        save_symbol(inputs['symbol'], inputs['time_window'])
    elif inputs['action'] == 'update':
        if inputs['symbol'] == None or inputs['time_window'] == None:
            print("must provide symbol and time window for update or save")
            exit()
        update_symbol(inputs['symbol'], inputs['time_window'])
    elif inputs['action'] == 'updateall':
        if inputs['time_window'] == None:
            print("Update all requires time window")
            exit()
        clear_and_get_full_data(inputs['time_window'])
    else:
        print("unknown option: {}".format(inputs['action']))


