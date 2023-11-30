import pandas as pd
import numpy as np
from datetime import datetime


def get_frame(forecast_results, indexes=[47, 51, 55, 71, 75, 79]):

    date_st = (datetime.today().strftime('%Y-%m-%d'))
    if len(indexes) == 6:
        hours = ['36 hour', '40 hour', '44 hour',
                 '60 hour', '64 hour', '68 hour']
    else:
        hours = np.nan

    df_sub_ec = pd.DataFrame({
        "forecast_date": date_st,
        "target": "energy",
        "horizon": hours,
        "q0.025": forecast_results.iloc[:, 0],
        "q0.25": forecast_results.iloc[:, 1],
        "q0.5": forecast_results.iloc[:, 2],
        "q0.75": forecast_results.iloc[:, 3],
        "q0.975": forecast_results.iloc[:, 4]}).reset_index()

    return df_sub_ec
