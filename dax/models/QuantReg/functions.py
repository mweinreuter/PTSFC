import numpy as np
import pandas as pd


def get_intraweek_vol(daxdf):

    daxdata = daxdf.copy()
    daxdata['Intraday_Volatility'] = daxdata['LogRetLag1'].apply(
        lambda x: x**2)
    daxdata['Intraweek_Volatility'] = daxdata['Intraday_Volatility'].rolling(
        window=5).mean()

    weeklyvolatilityvalue = np.nan
    daxdata['weekday'] = daxdata.index.weekday
    for index, row in daxdata.iterrows():
        if row['weekday'] == 2:
            weeklyvolatilityvalue = row['Intraweek_Volatility']
        daxdata.loc[index, 'intraweek_vol'] = weeklyvolatilityvalue

    daxdata = daxdata[6:].drop(
        columns=['Intraday_Volatility', 'Intraweek_Volatility', 'weekday'])

    return daxdata
