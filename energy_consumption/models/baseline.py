import pandas as pd
import numpy as np

from energy_consumption.feature_selection.extract import extract_energy_data

from datetime import datetime, date, timedelta


def get_baseline_forecasts(energydata=np.nan, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], indexes=[47, 51, 55, 71, 75, 79], abs_eval=False):

    if type(energydata) == float:
        energydata = extract_energy_data.get_data(
            num_years=5, set_wed=False)  # change to 5

    if len(energydata) > 43800:
        energydata = energydata[-43800:]

    energydata = energydata.rename(
        columns={"energy_consumption": "gesamt"})

    energydata["weekday"] = energydata.index.weekday
    horizons_def = indexes
    horizons = [h+1 for h in horizons_def]

    LAST_IDX = -1
    LAST_DATE = energydata.iloc[LAST_IDX].name

    horizon_date = [get_date_from_horizon(LAST_DATE, h) for h in horizons]

    # rows correspond to horizon, columns to quantile level
    pred_baseline = np.zeros((6, 5))

    last_t = 100

    # create df
    col_names = []
    for q in quantiles:
        col_names.append(f'q{q}')
    forecasts = pd.DataFrame(columns=col_names)

    for i, d in enumerate(horizon_date):

        weekday = d.weekday()
        hour = d.hour

        df_tmp = energydata.iloc[:LAST_IDX]

        cond = (df_tmp.weekday == weekday) & (df_tmp.index.time == d.time())

        new_row = np.quantile(
            df_tmp[cond].iloc[-last_t:]["gesamt"], q=quantiles)

        forecasts.loc[len(forecasts)] = new_row

    forecasts.index = horizon_date
    forecasts.index.name = 'date_time'

    if abs_eval == True:
        return forecasts

    else:
        date_str = datetime.today().strftime('%Y%m%d')

        date_str = date.today()  # - timedelta(days=1)
        date_str = date_str.strftime('%Y-%m-%d')

        horizons_adj = list(np.array(horizons_def) - 11)

        df_sub = pd.DataFrame({
            "forecast_date": date_str,
            "target": "energy",
            "horizon": [str(h) + " hour" for h in horizons_adj],
            "q0.025": forecasts['q0.025'],
            "q0.25": forecasts['q0.25'],
            "q0.5": forecasts['q0.5'],
            "q0.75": forecasts['q0.75'],
            "q0.975": forecasts['q0.975']})

        return df_sub


def get_date_from_horizon(last_ts, horizon):
    return last_ts + pd.DateOffset(hours=horizon)
