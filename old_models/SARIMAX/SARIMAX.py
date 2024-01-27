import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

from datetime import datetime

from old_models.extract_SARIMAX_features import get_energy_and_SARIMAX_features
from energy_consumption.models.SARIMAX.get_SARIMA_residuals import get_residuals


def get_SARIMAX_forecasts(energydata=pd.DataFrame, submission=True, periods=100, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):

    energydata = get_energy_and_SARIMAX_features(energydata)

    model = sm.tsa.SARIMAX(energydata['energy_consumption'], exog=energydata[[
        'sun_hours', 'saturday', 'working_day', 'holiday', 'index']],
        order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), approximate_diffuse=True)
    results = model.fit()

    # forecast means
    forecast = results.get_forecast(
        steps=periods, exog=energydata[['sun_hours', 'saturday', 'working_day', 'holiday', 'index']][-100:])
    forecast_mean = forecast.predicted_mean

    # forecast std proxy (abs. values of residuals)
    forecasts = get_residuals(energydata, forecast_mean, periods)

    # calculate confidence intervals
    for q in quantiles:
        forecasts[f'q{q}'] = forecasts['mean_forecast'] + \
            forecasts['residuals']*norm.ppf(q, loc=0)

    if submission == True:

        # Extract rows for specific weekdays and hours (make sure they are actually correct)
        friday_12 = forecasts[(forecasts.index.weekday == 4) & (
            forecasts.index.hour == 12)]
        friday_16 = forecasts[(forecasts.index.weekday == 4) & (
            forecasts.index.hour == 16)]
        friday_20 = forecasts[(forecasts.index.weekday == 4) & (
            forecasts.index.hour == 20)]

        saturday_12 = forecasts[(forecasts.index.weekday == 5) & (
            forecasts.index.hour == 12)]
        saturday_16 = forecasts[(forecasts.index.weekday == 5) & (
            forecasts.index.hour == 16)]
        saturday_20 = forecasts[(forecasts.index.weekday == 5) & (
            forecasts.index.hour == 20)]

        # Create a common DataFrame to store the results
        forecasts = pd.concat(
            [friday_12, friday_16, friday_20, saturday_12, saturday_16, saturday_20])

        date_st = (datetime.today().strftime('%Y-%m-%d'))
        hours = ['36 hour', '40 hour', '44 hour',
                 '60 hour', '64 hour', '68 hour']

        forecasts = forecasts.drop(columns=['mean_forecast', 'residuals'])
        forecasts.insert(0, 'forecast_date', date_st)
        forecasts.insert(1, 'target', 'energy')
        forecasts.insert(2, 'horizon', hours)

        forecasts = forecasts.reset_index().rename(
            columns={'index': 'date_time'})

    return (forecasts)
