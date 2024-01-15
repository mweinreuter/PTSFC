import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame
from energy_consumption.models.lasso.functions import get_interaction_and_pol_terms, estimate_forecast_std, get_quantiles


def get_Lasso_forecasts(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79],
                        quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], periods=100):

    if energydata.empty:
        energydata = extract_energy_data.get_data(num_years=1.73)

    # get standardized features
    if len(energydata) > 17520:
        energydata = extract_all_features.get_energy_and_standardized_features(
            energydata, lasso=True)[-17520:]
    else:
        energydata = extract_all_features.get_energy_and_standardized_features(
            energydata, lasso=True)

    # split df
    y = energydata[['energy_consumption']]
    X = energydata.drop(columns=['energy_consumption'])
    X = sm.add_constant(X, has_constant="add")

    # create dataframe to store forecast quantiles
    X_fc = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])
    X_fc = extract_all_features.get_energy_and_standardized_features(
        X_fc, lasso=True)
    X_fc = sm.add_constant(X_fc, has_constant='add')

    # approximate sun_hours by last observation
    # last_obs = len(X_int_pol) - 1
    # sun_hours_val = float(X_int_pol.iloc[last_obs][['sun_hours']])
    # X_fc['sun_hours'] = sun_hours_val

    X_int_pol = get_interaction_and_pol_terms(X)
    X_fc_int_pol = get_interaction_and_pol_terms(X_fc)

    # fit Lasso Regression with best alpha
    lasso = Lasso(alpha=0.0531)

    # Fit the model on the scaled data
    lasso.fit(X_int_pol, y)

    # estimate forecast means
    mean_est = lasso.predict(X_fc_int_pol)[indexes]

    # estimate forecast stds
    mean_est_hist = lasso.predict(X_int_pol)
    variance_est = mean_squared_error(
        y, mean_est_hist)
    indexes_shiftforw = [index+1 for index in indexes]
    forecast_std = np.array([estimate_forecast_std(
        variance_est, horizon) for horizon in indexes_shiftforw])

    # estimate quantile forecasts
    quantile_forecasts = get_quantiles(
        mean_est, forecast_std, quantiles)

    # return quantile forecasts in terms of absolute evaluation
    abs_eval = len(quantiles) != 6
    if abs_eval == True:
        horizon = pd.date_range(start=energydata.index[-1] + pd.DateOffset(
            hours=1), periods=periods, freq='H')
        quantile_forecasts.insert(
            0, 'date_time', [horizon[i] for i in indexes])

        return quantile_forecasts

    # else: create submission frame
    else:
        forecast_frame = create_submission_frame.get_frame(
            quantile_forecasts, indexes)
        forecast_frame = forecast_frame.drop(columns={'index'})
        horizon = pd.date_range(start=energydata.index[-1] + pd.DateOffset(
            hours=1), periods=periods, freq='H')
        forecast_frame.insert(
            0, 'date_time', [horizon[i] for i in indexes])

        return forecast_frame
