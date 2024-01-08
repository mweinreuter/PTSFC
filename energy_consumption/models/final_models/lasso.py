import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from energy_consumption.help_functions import get_energy_data, dummy_mapping, get_forecast_timestamps, lasso_functions, create_submission_frame
from energy_consumption.feature_selection.extract.extract_lasso_features import get_energy_and_features


def get_lasso_forecasts(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79],
                        quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], periods=100):

    if energydata.empty:
        energydata = get_energy_data.get_data()

    # get dummies
    energydata = get_energy_and_features(energydata)

    # split data to get observations
    y_obs = energydata['energy_consumption']
    X_obs = energydata.drop(columns=['energy_consumption'])
    X_obs = sm.add_constant(X_obs, has_constant="add")

    # fit Lasso Regression
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_obs, y_obs)

    # create dataframe to store forecast quantiles
    X_fc = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1], periods=periods)
    X_fc = get_energy_and_features(X_fc)
    X_fc = sm.add_constant(X_fc, has_constant='add')

    # estimate forecast means
    mean_est = lasso.predict(X_fc)[indexes]

    # estimate forecast stds
    mean_est_hist = lasso.predict(X_obs)
    variance_est = mean_squared_error(
        y_obs, mean_est_hist)
    indexes_shiftforw = [index+1 for index in indexes]
    forecast_std = np.array([lasso_functions.estimate_forecast_std(
        variance_est, horizon) for horizon in indexes_shiftforw])

    # estimate quantile forecasts
    quantile_forecasts = lasso_functions.get_quantiles(
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

        return (forecast_frame)
