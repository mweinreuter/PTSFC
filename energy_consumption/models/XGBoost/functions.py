import numpy as np
import pandas as pd

from energy_consumption.help_functions import get_forecast_timestamps


def get_opt_parameters(quantiles):

    quantile_params = dict()

    for alpha in quantiles:
        # set hyperparameters based on the best results for each alpha
        if alpha <= 0.1:
            params = dict({'n_estimators': 700, 'min_samples_split': 11,
                          'min_samples_leaf': 10, 'max_depth': 5, 'learning_rate': 0.3})
        elif alpha <= 0.35:
            params = dict({'n_estimators': 300, 'min_samples_split': 9,
                          'min_samples_leaf': 11, 'max_depth': 4, 'learning_rate': 0.2})
        elif alpha <= 0.65:
            params = dict({'n_estimators': 500, 'min_samples_split': 10,
                          'min_samples_leaf': 6, 'max_depth': 4, 'learning_rate': 0.1})
        elif alpha < 0.9:
            params = dict({'n_estimators': 500, 'min_samples_split': 10,
                          'min_samples_leaf': 6, 'max_depth': 4, 'learning_rate': 0.1})
        elif alpha >= 0.9:
            params = dict({'n_estimators': 200, 'min_samples_split': 7,
                          'min_samples_leaf': 7, 'max_depth': 5, 'learning_rate': 0.3})

        quantile_params[alpha] = params

    return quantile_params


def get_energy_and_forecast(energydata):

    energydf = energydata.copy()
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydf.index[-1])
    energyforecast['energy_consumption'] = np.nan
    merged = pd.concat([energydf, energyforecast]).reset_index()

    merged["hour"] = merged["date_time"].dt.hour
    merged["day_of_week"] = merged["date_time"].dt.dayofweek
    merged['weekly_lag'] = merged['energy_consumption'].shift(168)
    merged['yearly_lag'] = merged['energy_consumption'].shift(8760)
    merged = merged[-2260:]
    merged['index'] = range(1, len(merged) + 1)
    merged = merged.set_index('date_time')

    energydf = merged[-2260:-100]
    energyforecast = merged[-100:].drop(columns=['energy_consumption'])

    return energydf, energyforecast
