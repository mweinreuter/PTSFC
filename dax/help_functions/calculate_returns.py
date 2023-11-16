import numpy as np


def calculate_returns(daxdata, lags=1):

    for lag in list(range(1, lags+1)):
        calculate_return(daxdata, lag)

    # drop rows containing NaNs
    daxdata = daxdata.iloc[lags:, :]

    return daxdata


def calculate_return(daxdata, lag=1):

    daxdata[f'CloseLag{lag}'] = daxdata['Close'].shift(lag)
    daxdata[f'RetLag{lag}'] = 100*(daxdata['Close']-daxdata[f'CloseLag{lag}'])
    daxdata[f'LogRetLag{lag}'] = 100 * \
        (np.log(daxdata['Close']) - np.log(daxdata[f'CloseLag{lag}']))

    return daxdata
