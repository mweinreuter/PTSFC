import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta

from evaluation.help_functions.prepare_data import most_recent_thursday
from dax.help_functions.calculate_returns import calculate_returns


def get_data():

    msft = yf.Ticker("^GDAXI")
    daxdata = msft.history(period="max")

    return (daxdata)


def get_prepared_data(daxdata=pd.DataFrame):

    if daxdata.empty:
        daxdata = get_data()

    daxdata = calculate_returns(daxdata, lags=5)
    start_date_excl = most_recent_thursday(daxdata) - timedelta(days=1)
    daxdata = daxdata.loc[(daxdata.index >= daxdata.index[8000])
                          & (daxdata.index < start_date_excl)]
    daxdata = daxdata[[
        'Close', 'LogRetLag1', 'LogRetLag2', 'LogRetLag3', 'LogRetLag4', 'LogRetLag5']]
    daxdata = daxdata.dropna()

    return (daxdata)
