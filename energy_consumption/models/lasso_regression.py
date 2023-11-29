import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from energy_consumption.help_functions import get_energy_data, dummy_mapping, get_forecast_timestamps, lasso_functions, create_submission_frame


def get_lasso_forecasts(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79]):

    if energydata.empty:
        energydata = get_energy_data.get_data()

    # get dummies
    energydata = dummy_mapping.get_mappings(energydata)

    # split data
    y_ec = energydata['energy_consumption']
    X_ec = energydata.drop(
        columns=['energy_consumption'])
    X_ec = sm.add_constant(X_ec, has_constant="add")

    # include interaction terms
    poly_input = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interaction = poly_input.fit_transform(X_ec)

    # fit Lasso Regression
    lasso = Lasso(alpha=0.001)
    lasso.fit(X_interaction, y_ec)

    # create dataframe to store forecast quantiles
    X_fc = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])
    X_fc = dummy_mapping.get_mappings(X_fc)
    X_fc = sm.add_constant(X_fc, has_constant='add')
    poly_forecast = PolynomialFeatures(
        interaction_only=True, include_bias=False)
    X_fc_interaction = poly_forecast.fit_transform(X_fc)

    # estimate forecast means
    y_pred = lasso.predict(X_fc_interaction)
    selected_forecasts = y_pred[indexes]

    # estimate forecast stds
    pred_historical = lasso.predict(X_interaction)
    model_variance_est = mean_squared_error(
        y_ec, pred_historical)
    forecast_std = np.array([lasso_functions.estimate_forecast_std(
        model_variance_est, horizon) for horizon in [48, 52, 56, 72, 76, 80]])

    # estimate quantile forecasts
    quantile_forecasts = lasso_functions.get_quantiles(
        selected_forecasts, forecast_std)

    forecast_frame = create_submission_frame.get_frame(
        quantile_forecasts)
    forecast_frame = forecast_frame.drop(columns={'index'})

    # set horizon for the next 5 days
    horizon = pd.date_range(start=energydata.index[-1] + pd.DateOffset(
        hours=1), periods=90, freq='H')
    forecast_frame.insert(
        0, 'date_time', [horizon[i] for i in [47, 51, 55, 71, 75, 79]])

    return (forecast_frame)
