import pandas as pd
import numpy as np

import statsmodels.api as sm

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame
from energy_consumption.models.quantreg.functions import get_energy_and_forecast_quantreg


def get_QuantRegExLags_forecasts(energydf=np.nan, indexes=[47, 51, 55, 71, 75, 79], quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], abs_eval=False):

    if type(energydf) == float:
        energydf = extract_energy_data.get_data(num_years=2)

    energydata = energydf.copy()
    energydata, X_pred = get_energy_and_forecast_quantreg(energydata)

    X = energydata.drop(columns=['energy_consumption'])
    y = energydata['energy_consumption']

    # model
    model_qr = sm.QuantReg(y, X)

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])

    for q in quantiles:
        model_temp = model_qr.fit(q=q)
        forecast_temp = model_temp.predict(X_pred)
        energyforecast[f'q{q}'] = forecast_temp

    first_name = f'q{quantiles[0]}'
    max_index = len(quantiles) - 1
    last_name = f'q{quantiles[max_index]}'

    selected_forecasts = energyforecast.loc[energyforecast.index[indexes],
                                            first_name:last_name]

    if abs_eval == False:
        selected_forecasts = create_submission_frame.get_frame(
            selected_forecasts)

    return selected_forecasts
