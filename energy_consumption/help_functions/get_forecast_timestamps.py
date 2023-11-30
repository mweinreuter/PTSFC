import pandas as pd


def forecast_timestamps(last_ts, periods=100):

    # set horizon for the next 5 days
    horizon = pd.date_range(start=last_ts + pd.DateOffset(
        hours=1), periods=periods, freq='H')

    energyforecast = pd.DataFrame({'date_time': horizon})
    energyforecast.set_index('date_time', inplace=True)

    return energyforecast
