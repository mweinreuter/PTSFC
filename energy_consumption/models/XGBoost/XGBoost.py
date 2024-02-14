import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from energy_consumption.feature_selection.extract import extract_energy_data
from energy_consumption.help_functions import create_submission_frame
from energy_consumption.models.XGBoost.functions import get_energy_and_forecast, get_opt_parameters


def get_XGBoost_forecasts(energydf=np.nan, indexes=[47, 51, 55, 71, 75, 79], quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], periods=100, abs_eval=False):

    if type(energydf) == float:
        energydf = extract_energy_data.get_data(num_years=2)

    energydata = energydf.copy()
    energydata, X_pred = get_energy_and_forecast(energydata)

    X = energydata.drop(columns=['energy_consumption'])
    y = energydata['energy_consumption']

    quantile_df = pd.DataFrame()
    quantile_params = get_opt_parameters(quantiles)

    for alpha in quantiles:
        name = f'q{alpha}'
        opt_params = quantile_params[alpha]
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, **opt_params)
        quantile_model = gbr.fit(X, y)
        y_pred = quantile_model.predict(X_pred)
        quantile_df[name] = y_pred

    quantile_df = quantile_df.iloc[indexes]

    quantile_cols = [f'q{q}' for q in quantiles]

    # Loop through the quantiles
    for i in range(len(quantile_cols) - 1):
        current_quantile = quantile_cols[i]
        next_quantile = quantile_cols[i + 1]

        # Check if there is an overlap
        overlap = quantile_df[current_quantile] > quantile_df[next_quantile]

        # Adjust the values if there is an overlap
        quantile_df.loc[overlap,
                        next_quantile] = quantile_df[current_quantile][overlap]

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
