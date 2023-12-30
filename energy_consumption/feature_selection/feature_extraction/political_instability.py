
# finished --> no adjustments needed

import yfinance as yf
import numpy as np
import pandas as pd
from dax.help_functions.calculate_returns import calculate_return


def calculate_dax_means(energydata):

    # import daxdata accordingly to energydaata
    # offset due to lag and mean calculations
    first_timestamp = energydata.index.min() - pd.DateOffset(days=8)
    last_timestamp = energydata.index.max()

    msft = yf.Ticker("^GDAXI")
    daxdata = msft.history(start=first_timestamp, end=last_timestamp)

    # calculate returns and weekly means
    daxdata = calculate_return(daxdata)[['Close', 'LogRetLag1']]
    daxdata['AbsLogRetLag1'] = daxdata['LogRetLag1'].abs()
    daxdata['CloseMean5Days'] = daxdata['Close'].rolling(window=5).mean()
    daxdata['AbsLogRetLag1Mean5Days'] = daxdata['AbsLogRetLag1'].rolling(
        window=5).mean()

    # store weekly means for every week starting by wednesday
    current_closevalue = np.nan
    current_abslogretvalue = np.nan
    daxdata['Weekday'] = daxdata.index.weekday

    for index, row in daxdata.iterrows():
        if row['Weekday'] == 2:
            current_closevalue = row['CloseMean5Days']
            current_abslogretvalue = row['AbsLogRetLag1Mean5Days']

        daxdata.loc[index, 'close_weekly'] = current_closevalue
        daxdata.loc[index, 'abs_log_ret_weekly'] = current_abslogretvalue

    # drop NaNs and select columns
    daxdata = daxdata.dropna(subset=['close_weekly', 'abs_log_ret_weekly',
                                     'LogRetLag1', 'AbsLogRetLag1', 'CloseMean5Days',
                                     'AbsLogRetLag1Mean5Days'])[['close_weekly', 'abs_log_ret_weekly']]

    return daxdata


def ec_dax_merge(energydata, daxdata=pd.DataFrame):

    if daxdata.empty:
        daxdata = calculate_dax_means(energydata)

    # prepare for merge
    energydata['date'] = energydata.index.date
    energydata = energydata.reset_index()
    daxdata['date'] = daxdata.index.date

    # merge data
    energy_merged = pd.merge(daxdata, energydata, how='left', on='date').set_index(
        'date_time').drop(columns={'date'})

    return (energy_merged)
