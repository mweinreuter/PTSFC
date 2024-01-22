import numpy as np
import pandas as pd

from energy_consumption.feature_selection.extract import extract_energy_data
from energy_consumption.models.combination.combine_models import get_combined_models

# import required models
from energy_consumption.models.lasso.lasso import get_Lasso_forecasts
from energy_consumption.models.quantreg.quantreg import get_QuantReg_forecasts
from old_models.baseline import get_baseline_forecasts
from energy_consumption.models.knn.knn import get_KNNRegression_forecasts
from energy_consumption.models.xgboost.XGBoost import get_XGBoost_forecasts


def get_combined_DEMO(energydata=np.nan):

    if type(energydata) == float:
        # use derived optimum for number of years
        energydata = extract_energy_data.get_data(num_years=7)

    # make sure to import energydata before
    lasso = {
        "name": "lasso",
        "function": get_Lasso_forecasts,
        "horizon_integration": [True, True, True, False, False, False],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    quantReg = {
        "name": "quantile regression",
        "function": get_QuantReg_forecasts,
        "horizon_integration": [True, True, True, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    baseline = {
        "name": "baseline",
        "function": get_baseline_forecasts,
        "horizon_integration": [False, False, False, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    knn = {
        "name": "KNNRegression",
        "function": get_KNNRegression_forecasts,
        "horizon_integration": [False, False, False, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    XGBoost = {
        'name': 'xgboost',
        'function': get_XGBoost_forecasts,
        "horizon_integration": [False, False, False, True, True, True],
        "weights": [1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
    }

    models = [lasso, quantReg, baseline, knn, XGBoost]
    combined_model = get_combined_models(
        models, energydata)

    return combined_model
