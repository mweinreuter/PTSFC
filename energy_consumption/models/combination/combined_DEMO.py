import numpy as np
import pandas as pd

from energy_consumption.feature_selection.extract import extract_energy_data
from energy_consumption.models.combination.combine_models import get_combined_models

# import required models

from energy_consumption.models.lasso.lasso import get_Lasso_forecasts
from energy_consumption.models.quantreg.quantreg_cali_short import get_QuantReg_forecasts
from old_models.XGBoost import get_XGBoost_forecasts
from energy_consumption.models.quantreg.quant_reg_short import get_QuantRegShort_forecasts


def get_combined_DEMO(energydata=np.nan):

    if type(energydata) == float:
        # use derived optimum for number of years
        energydata = extract_energy_data.get_data(num_years=7)

    # make sure to import energydata before
    lasso = {
        "name": "lasso",
        "function": get_Lasso_forecasts,
        "horizon_integration": [False, False, False, False, True, False],
        "weights": [0, 0, 0, 0, 1/2, 0]
    }

    quantReg = {
        "name": "quantile regression cali",
        "function": get_QuantReg_forecasts,
        "horizon_integration": [True, True, False, True, True, False],
        "weights": [1/2, 1/2, 0, 1/2, 1/2, 0]
    }

    quantregshort = {
        "name": "quantile regression short",
        "function": get_QuantRegShort_forecasts,
        "horizon_integration": [True, True, False, False, False, False],
        "weights": [1/2, 1/2, 0, 0, 0, 0]
    }

    XGBoost = {
        'name': 'xgboost',
        'function': get_XGBoost_forecasts,
        "horizon_integration": [False, False, True, False, True, True],
        "weights": [0, 0, 1, 1/2, 0, 1]
    }

    models = [lasso, quantReg, quantregshort, XGBoost]
    combined_model = get_combined_models(
        models, energydata)

    return combined_model
