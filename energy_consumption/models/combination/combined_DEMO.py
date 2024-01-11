import numpy as np
import pandas as pd

from energy_consumption.help_functions.get_energy_data import get_data
from energy_consumption.models.combination.combine_models import get_combined_models

# make sure that required models are imported
from energy_consumption.models.time_series_models.QR_mappings_interactions import get_QR_mappings_interactions
from energy_consumption.models.quantreg.quantreg import get_QuantReg_forecasts
from energy_consumption.models.time_series_models.seasonal_QR_hh import get_seasonal_QR_hourly_holidays


def get_combined_DEMO(energydata=pd.DataFrame, wednesday_morning=False):

    if energydata.empty:
        energydata = get_data(wednesday_morning=wednesday_morning)

    # make sure to import energydata before
    model_km = {
        "name": "km",
        "function": get_QR_mappings_interactions,
        "horizon_integration": [True, True, True, False, False, False],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    model_QuantReg = {
        "name": "QuantReg",
        "function": get_QuantReg_forecasts,
        "horizon_integration": [True, True, True, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    model_disaggregated = {
        "name": "disagg",
        "function": get_seasonal_QR_hourly_holidays,
        "horizon_integration": [False, False, False, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    models = [model_km, model_QuantReg, model_disaggregated]
    combined_model = get_combined_models(
        models, energydata, wednesday_morning=True)

    return combined_model
