import pandas as pd
import numpy as np

from datetime import datetime, date, timedelta
from energy_consumption.help_functions import get_energy_data


def get_baseline_forecasts(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79],
                           quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], periods=100):

    if energydata.empty:
        energydata = get_energy_data.get_data()

    energydata = energydata.rename(columns={"energy_consumption": "gesamt"})
    energydata["weekday"] = energydata.index.weekday

    LAST_IDX = -1
    LAST_DATE = energydata.iloc[LAST_IDX].name

    horizon_date = [get_date_from_horizon(LAST_DATE, i) for i in indexes]

    # rows correspond to horizon, columns to quantile level
    pred_baseline = np.zeros((6, 5))
    last_t = 100

    for i, d in enumerate(horizon_date):

        weekday = d.weekday()
        hour = d.hour

        df_tmp = energydata.iloc[:LAST_IDX]

        cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())

        pred_baseline[i, :] = np.quantile(
            df_tmp[cond].iloc[-last_t:]["gesamt"], q=quantiles)

    date_str = datetime.today().strftime('%Y%m%d')
    horizons_def = [36, 40, 44, 60, 64, 68]

    df_sub = pd.DataFrame({
        "forecast_date": date_str,
        "target": "energy",
        "horizon": [str(h) + " hour" for h in horizons_def],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]})

    return (df_sub)


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)
