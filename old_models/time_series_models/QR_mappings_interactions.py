import pandas as pd
import numpy as np

import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures

from energy_consumption.help_functions import get_energy_data, dummy_mapping, get_forecast_timestamps, create_submission_frame
from old_models import handle_outstanding_dp

# proper time mapping and deletion of outstanding data points
# no need to adjust day and season mapping (vizualisations sufficient)


def get_QR_mappings_interactions(energydata=pd.DataFrame(), indexes=[47, 51, 55, 71, 75, 79]):

    if energydata.empty:
        energydata = get_energy_data.get_data()

    # get dummies
    energydata = dummy_mapping.get_mappings(energydata)

    # quantile regression data
    y_ec = energydata['energy_consumption']
    X_ec = energydata.drop(
        columns=['energy_consumption'])
    X_ec = sm.add_constant(X_ec, has_constant="add")

    # include interaction terms
    poly_input = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_interaction = poly_input.fit_transform(X_ec)

    # create dataframe to store forecast quantiles
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydata.index[-1])
    energyforecast = dummy_mapping.get_mappings(energyforecast)
    X_fc = sm.add_constant(energyforecast, has_constant='add')

    poly_forecast = PolynomialFeatures(
        interaction_only=True, include_bias=False)
    X_fc_interaction = poly_forecast.fit_transform(X_fc)

    # model
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    model_qr = sm.QuantReg(y_ec, X_interaction)

    for q in quantiles:
        model_temp = model_qr.fit(q=q)
        forecast_temp = model_temp.predict(X_fc_interaction)
        energyforecast[f'forecast{q}'] = forecast_temp

    selected_forecasts = energyforecast.loc[energyforecast.index[indexes],
                                            'forecast0.025':'forecast0.975']
    selected_forecasts_frame = create_submission_frame.get_frame(
        selected_forecasts)

    return (selected_forecasts_frame)
