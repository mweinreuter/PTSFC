import pandas as pd
import numpy as np

import statsmodels.api as sm

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame


def get_QuantRegFS_forecasts(energydata=np.nan, indexes=[47, 51, 55, 71, 75, 79], quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], abs_eval=False):

    if type(energydata) == float:
        # use derived optimum for number of years (see notebook)
        energydata = extract_energy_data.get_data(num_years=6.17)

    # get features
    if len(energydata) > 54027:
        energydata = extract_all_features.get_energy_and_features(energydata,
                                                                  feature_selection=True)[-54027:]
    else:
        energydata = extract_all_features.get_energy_and_features(energydata,
                                                                  feature_selection=True)

    X = energydata.drop(columns=[
                        'energy_consumption', 'population', 'spring_autumn', 'abs_log_ret_weekly'])
    X.insert(loc=0, column='constant', value=1)
    y = energydata['energy_consumption']

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])

    X_pred = extract_all_features.get_energy_and_features(energyforecast,
                                                          feature_selection=True)
    X_pred = X_pred.drop(
        columns=['population', 'spring_autumn', 'abs_log_ret_weekly'])
    X_pred.insert(loc=0, column='constant', value=1)

    # model
    model_qr = sm.QuantReg(y, X)

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
