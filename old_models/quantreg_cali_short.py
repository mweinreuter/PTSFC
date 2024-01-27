import pandas as pd
import numpy as np

from scipy.stats import norm

import statsmodels.api as sm

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame


def get_QuantReg_forecasts(energydata=np.nan, indexes=[47, 51, 55, 71, 75, 79], quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], abs_eval=False):

    if type(energydata) == float:
        # use derived optimum for number of years (see notebook)
        energydata = extract_energy_data.get_data(num_years=0.25)

    # get features
    if len(energydata) > 1000:
        energydata = extract_all_features.get_energy_and_features(energydata,
                                                                  feature_selection_comp=True)[-1000:]
    else:
        energydata = extract_all_features.get_energy_and_features(energydata,
                                                                  feature_selection_comp=True)

    # new: drop index und winter, since they are not important for monthly forecasts
    X = energydata.drop(columns=['energy_consumption'])
    X.insert(loc=0, column='constant', value=1)
    y = energydata['energy_consumption']

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])

    X_pred = extract_all_features.get_energy_and_features(energyforecast,
                                                          feature_selection_comp=True)
    X_pred.insert(loc=0, column='constant', value=1)

    # model
    model_qr = sm.QuantReg(y, X)

    for q in quantiles:
        model_temp = model_qr.fit(q=q)
        forecast_temp = model_temp.predict(X_pred)
        energyforecast[f'q{q}'] = forecast_temp + norm.ppf(q, loc=0)*2.5

    first_name = f'q{quantiles[0]}'
    max_index = len(quantiles) - 1
    last_name = f'q{quantiles[max_index]}'

    selected_forecasts = energyforecast.loc[energyforecast.index[indexes],
                                            first_name:last_name]
    if abs_eval == False:
        selected_forecasts = create_submission_frame.get_frame(
            selected_forecasts)

    return selected_forecasts
