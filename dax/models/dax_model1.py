import pandas as pd
import numpy as np
from datetime import datetime

from dax.help_functions.get_dax_data import get_data
from dax.help_functions.calculate_returns import calculate_returns


def get_dax_forecasts_model1(daxdata=pd.DataFrame(), last_t=1000):

    if daxdata.empty:
        daxdata = get_data()
        # calculate log returns
        daxdata = calculate_returns(daxdata, lags=5)

    # quantile levels
    tau = [.025, .25, .5, .75, .975]

    # define prediction array
    # cols are quantile levels, rows are horizons
    pred_baseline = np.zeros((5, 5))

    for i in range(5):
        ret_str = 'LogRetLag'+str(i+1)
        # Check if the slicing result is a NumPy array
        sliced_array = daxdata[ret_str].iloc[-last_t:].values

        # Check for NaN values
        if np.isnan(sliced_array).any():
            print(
                f"Warning: NaN values found in sliced_array for ret_str: {ret_str}")
        else:
            pred_baseline[i, :] = np.quantile(
                daxdata[ret_str].iloc[-last_t:].values, q=tau)

    # create submission table
    dax_forecasts = pd.DataFrame({
        "target": "DAX",
        "horizon": [str(i) + " day" for i in (1, 2, 5, 6, 7)],
        "q0.025": pred_baseline[:, 0],
        "q0.25": pred_baseline[:, 1],
        "q0.5": pred_baseline[:, 2],
        "q0.75": pred_baseline[:, 3],
        "q0.975": pred_baseline[:, 4]})

    # define date_times for horizons
    last_date = daxdata.index[-1]
    first_index_date = last_date + pd.Timedelta(days=1)
    dax_forecasts.index = pd.date_range(
        start=first_index_date, periods=5, freq='B')

    date_st = (datetime.today().strftime('%Y-%m-%d'))
    dax_forecasts.insert(0, 'forecast_date', date_st)

    return (dax_forecasts)
