import pandas as pd
import numpy as np

import statsmodels.api as sm

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.feature_selection.collect.dummy_mapping import get_mappings_baseline
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame


def get_baseline_forecasts(energydata=np.nan, indexes=[47, 51, 55, 71, 75, 79]):

    if type(energydata) == float:

        # use derived optimum for number of years
        energydata = extract_energy_data.get_data(num_years=7)

    energydata = get_mappings_baseline(energydata)

    X = energydata.drop(columns=['energy_consumption'])
    X = sm.add_constant(X, has_constant="add")
    y = energydata['energy_consumption']

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])

    energyforecast = get_mappings_baseline(energyforecast)
    X_pred = sm.add_constant(energyforecast, has_constant='add')

    # make sure predictors have same dimension
    X_pred_all = pd.DataFrame()
    for col in X.columns:
        if col in X_pred.columns:
            X_pred_all[col] = X_pred[col]
        else:
            X_pred_all[col] = 0

        # model
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    model_qr = sm.QuantReg(y, X)

    for q in quantiles:
        model_temp = model_qr.fit(q=q)
        forecast_temp = model_temp.predict(X_pred_all)
        energyforecast[f'forecast{q}'] = forecast_temp

    selected_forecasts = energyforecast.loc[energyforecast.index[indexes],
                                            'forecast0.025':'forecast0.975']
    selected_forecasts_frame = create_submission_frame.get_frame(
        selected_forecasts)

    return selected_forecasts_frame
