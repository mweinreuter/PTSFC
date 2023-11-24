import pandas as pd
import numpy as np

import statsmodels.api as sm

from energy_consumption.help_functions import get_energy_data, dummy_mapping, get_forecast_timestamps, create_submission_frame


def get_seasonal_QR_hourly(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79]):

    if energydata.empty:
        energydata = get_energy_data.get_data()

    # select data since two years
    nrows = len(energydata)-17472
    energydata = energydata.iloc[nrows:]

    # create dummies
    energydata = dummy_mapping.get_season_mapping(energydata)
    energydata = dummy_mapping.get_day_mapping(energydata)
    energydata = dummy_mapping.get_hour_mapping(energydata)

    # quantile regression data
    y_ec = energydata['energy_consumption']
    X_ec = energydata.drop(
        columns=['energy_consumption'])
    X_ec = sm.add_constant(X_ec, has_constant="add")

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])
    energyforecast = dummy_mapping.get_season_mapping(energyforecast)
    energyforecast = dummy_mapping.get_day_mapping(energyforecast)
    energyforecast = dummy_mapping.get_hour_mapping(energyforecast)
    X_fc = sm.add_constant(energyforecast, has_constant='add')

    # model
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    model_qr = sm.QuantReg(y_ec, X_ec)

    for q in quantiles:
        model_temp = model_qr.fit(q=q)

        # Calculate forecasts for X_fc using the fitted model for the current quantile
        forecast_temp = model_temp.predict(X_fc)

        # Add the forecasts to the energy_forecast DataFrame with a label like 'forecast025'
        energyforecast[f'forecast{q}'] = forecast_temp

    selected_forecasts = energyforecast.loc[energyforecast.index[indexes],
                                            'forecast0.025':'forecast0.975']

    selected_forecasts_frame = create_submission_frame.get_frame(
        selected_forecasts)

    return (selected_forecasts_frame)
