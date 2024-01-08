import pandas as pd
import numpy as np
import statsmodels.api as sm

# calculate residuals for last number of periods using the same model


def get_residuals(energydata, forecast_mean, periods):

    energy_actual_train = energydata.iloc[:-periods]['energy_consumption']
    exog_actual_train = energydata.iloc[:-100][[
        'sun_hours', 'saturday', 'working_day', 'holiday', 'index']]
    model_st = sm.tsa.SARIMAX(energy_actual_train, exog=exog_actual_train,
                              order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), approximate_diffuse=True)
    results_st = model_st.fit()

    # past forecasts
    forecast_st = results_st.get_forecast(
        steps=periods, exog=energydata[['sun_hours', 'saturday', 'working_day', 'holiday', 'index']].iloc[-200:-100, :])

    predicted_mean = forecast_st.predicted_mean
    actual_mean = energydata.iloc[-100:, :]['energy_consumption']

    together = pd.DataFrame(
        {'forecast': predicted_mean, 'actual': actual_mean})
    together['residuals'] = abs(together['actual'] - together['forecast'])
    residuals = np.array(together['residuals'])

    forecasts = pd.DataFrame(
        {'mean_forecast': forecast_mean, 'residuals': residuals})

    return forecasts
