import numpy as np
import pandas as pd

from dax.models.ARMA_GARCH.arma_garch import get_arma_garch11_forecasts, get_arma_garch_opt_pq_forecasts, get_arma_garch_opt_pq_lag_forecasts
from dax.models.baseline import get_dax_forecasts_baseline
from dax.models.QuantReg.quantile_regression import get_quantile_regression_forecasts
from dax.models.QuantReg.QuantReg import get_QuantReg_forecasts, get_QuantReg_forecasts_abs
from dax.models.QuantReg.QuantRegFeatures import get_QuantRegFeatures_forecasts
from dax.models.ARMA_NN.arma_nn import get_arma_nn_forecasts

# only combination ob garch(1,1) and baseline possible (in any other case: change import statements)


def combine_models(daxdata=pd.DataFrame, weights=[1/5, 2/5, 2/5]):

    QuantRegAbs_forecasts = get_QuantReg_forecasts_abs(daxdata)
    QuantRegFeat_Forecasts = get_QuantRegFeatures_forecasts(daxdata)
    ARMA_nn_Forecasts = get_arma_nn_forecasts(daxdata)

    combined_model = QuantRegAbs_forecasts.copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    QuantRegAbs_forecasts_array = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'])
    QuantRegFeat_Forecasts_array = np.array(
        QuantRegFeat_Forecasts.loc[:, 'q0.025':'q0.975'])
    ARMA_nn_Forecasts_array = np.array(
        ARMA_nn_Forecasts.loc[:, 'q0.025':'q0.975'])

    combined_value_array = weights[0]*garch_t_forecast_array + \
        weights[1]*baseline_forecast_array + \
        weights[2]*quantile_regression_forecast_array
    combined_value_array

    for i in range(len(combined_model)):
        combined_model.loc[combined_model.index[i],
                           'q0.025':'q0.975'] = combined_value_array[i]

    return (combined_model)
