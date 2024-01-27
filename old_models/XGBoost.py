import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from energy_consumption.feature_selection.extract import extract_energy_data, extract_all_features
from energy_consumption.help_functions.drop_years import drop_years
from energy_consumption.help_functions import get_forecast_timestamps, create_submission_frame

optimized_params = dict(
    learning_rate=0.01,
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=13,
    min_samples_split=11,
)


def get_XGBoost_forecasts(energydata=np.nan, indexes=[47, 51, 55, 71, 75, 79], quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], periods=100, abs_eval=False):

    if type(energydata) == float:
        energydata = extract_energy_data.get_data(num_years=1.15)

    if len(energydata) > 10000:
        energydata = extract_all_features.get_energy_and_features(
            energydata, feature_selection_comp=True)[-10000:]
    else:
        energydata = extract_all_features.get_energy_and_features(
            energydata, feature_selection_comp=True)

    X = energydata.drop(columns=['energy_consumption'])
    y = energydata['energy_consumption']

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])

    X_pred = extract_all_features.get_energy_and_features(
        energyforecast, feature_selection_comp=True)

    X, X_pred = drop_years(X, X_pred)

    quantile_df = pd.DataFrame()
    for alpha in quantiles:
        name = f'q{alpha}'
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, **optimized_params)
        quantile_model = gbr.fit(X, y)
        y_pred = quantile_model.predict(X_pred)
        quantile_df[name] = y_pred

    quantile_df = quantile_df.iloc[indexes]

    # return quantile forecasts in terms of absolute evaluation
    if abs_eval == True:
        horizon = pd.date_range(start=energydata.index[-1] + pd.DateOffset(
            hours=1), periods=periods, freq='H')
        quantile_df.insert(
            0, 'date_time', [horizon[i] for i in indexes])

        return quantile_df

    # else: create submission frame
    else:
        forecast_frame = create_submission_frame.get_frame(
            quantile_df, indexes)
        forecast_frame = forecast_frame.drop(columns={'index'})
        horizon = pd.date_range(start=energydata.index[-1] + pd.DateOffset(
            hours=1), periods=periods, freq='H')
        forecast_frame.insert(
            0, 'date_time', [horizon[i] for i in indexes])

        return forecast_frame
