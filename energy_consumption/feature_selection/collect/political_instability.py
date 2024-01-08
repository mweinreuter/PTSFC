import yfinance as yf
import numpy as np
import pandas as pd
from dax.help_functions.calculate_returns import calculate_return


def calculate_dax_means(energydata):

    # import daxdata accordingly to energydaata
    # offset due to lag and mean calculations
    first_timestamp = energydata.index.min() - pd.DateOffset(days=9)
    last_timestamp = energydata.index.max()

    msft = yf.Ticker("^GDAXI")
    daxdata = msft.history(start=first_timestamp, end=last_timestamp)

    # calculate returns
    daxdata = calculate_return(daxdata)[['Close', 'LogRetLag1']]

    # initialize window size
    window_size = 5

    # calculate absolute returns
    daxdata['AbsLogRetLag1'] = daxdata['LogRetLag1'].abs()

    # calculate means
    daxdata['CloseMean5Days'] = daxdata['Close'].rolling(
        window=window_size).mean()
    daxdata['AbsLogRetLag1Mean5Days'] = daxdata['AbsLogRetLag1'].rolling(
        window=5).mean()

    # calculate close-to-close volatility (rolling standard deviation)
    daxdata['cc_volatility_weekly'] = daxdata['LogRetLag1'].rolling(
        window=window_size).std()

    # store weekly means and close-to-close-volatility for every week starting by wednesday
    closevalue = np.nan
    abslogretvalue = np.nan
    volatilityvalue = np.nan
    daxdata['Weekday'] = daxdata.index.weekday

    for index, row in daxdata.iterrows():
        if row['Weekday'] == 2:
            closevalue = row['CloseMean5Days']
            abslogretvalue = row['AbsLogRetLag1Mean5Days']
            volatilityvalue = row['cc_volatility_weekly']

        daxdata.loc[index, 'close_weekly'] = closevalue
        daxdata.loc[index, 'abs_log_ret_weekly'] = abslogretvalue
        daxdata.loc[index, 'volatility_weekly'] = volatilityvalue

    # drop NaNs and select columns
    daxdata = daxdata.dropna(subset=['close_weekly', 'abs_log_ret_weekly',
                                     'LogRetLag1', 'AbsLogRetLag1', 'CloseMean5Days',
                                     'AbsLogRetLag1Mean5Days', 'cc_volatility_weekly'])[['close_weekly', 'abs_log_ret_weekly', 'volatility_weekly']]

    return daxdata


def ec_dax_merge(energydata, daxdata=pd.DataFrame):

    if daxdata.empty:
        daxdata = calculate_dax_means(energydata)

    # prepare for merge
    energydata['date'] = energydata.index.date
    energydata = energydata.reset_index()
    daxdata['date'] = daxdata.index.date

    # merge data
    energy_merged = pd.merge(energydata, daxdata, how='left', on='date').set_index(
        'date_time').drop(columns={'date'})

    # Fill NaN values with the value of the previous row
    energy_merged[['close_weekly', 'abs_log_ret_weekly', 'volatility_weekly']] = energy_merged[['close_weekly',
                                                                                                'abs_log_ret_weekly', 'volatility_weekly']].fillna(method='ffill')

    return (energy_merged)
