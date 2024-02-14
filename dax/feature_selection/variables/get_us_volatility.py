import pandas as pd
import numpy as np
import yfinance as yf


def get_us_volatility_data(daxdata):

    msft = yf.Ticker("^VIX")
    vol_data = msft.history(period="max")

    us_vol_data = vol_data[['Close']].rename(
        columns={'Close': 'us_volatility'})
    us_vol_data['date'] = us_vol_data.index.date

    if 'date' not in daxdata.columns:
        daxdata['date'] = daxdata.index.date

    merged = pd.merge(daxdata, us_vol_data, how='left',
                      on='date').set_index('date')

    # calculate weekly means
    merged['weekly_mean_vol'] = merged['us_volatility'].rolling(
        window=5).mean()

    # store weekly means and close-to-close-volatility for every week starting by wednesday
    volatilityvalue = np.nan
    merged['weekday'] = daxdata.index.weekday

    for index, row in merged.iterrows():
        if row['weekday'] == 2:
            volatilityvalue = row['weekly_mean_vol']
        merged.loc[index, 'volatility_weekly'] = volatilityvalue

    print(len(merged))
    merged = merged[7:].drop(
        columns=['us_volatility', 'weekly_mean_vol', 'weekday']).dropna()

    print(len(merged))

    return merged
