import numpy as np
import pandas as pd

from dax.models.ARMA_GARCH.arma_garch import get_arma_garch11_forecasts, get_arma_garch_opt_pq_forecasts, get_arma_garch_opt_pq_lag_forecasts
from dax.models.QuantReg.QuantReg import get_QuantReg_forecasts_abs
from dax.models.QuantReg.QuantRegFeatures import get_QuantRegFeatures_forecasts
from dax.models.ARMA_NN.arma_nn import get_arma_nn_forecasts


def combine_models(daxdata=pd.DataFrame):

    QuantRegAbs_forecasts = get_QuantReg_forecasts_abs(daxdata)
    QuantRegFeat_Forecasts = get_QuantRegFeatures_forecasts(daxdata)
    ARMA_nn_Forecasts = get_arma_nn_forecasts(daxdata)
    Garch_pq = get_arma_garch_opt_pq_forecasts(daxdata)

    combined_model = QuantRegAbs_forecasts.copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    combined_model.loc[:, 'q0.025':'q0.975'].iloc[0] = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.7 + np.array(QuantRegFeat_Forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.3
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[1] = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.2 + np.array(QuantRegFeat_Forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.2
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[2] = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.5 + np.array(QuantRegFeat_Forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.5
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[3] = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*0.7 + np.array(Garch_pq.loc[:, 'q0.025':'q0.975'].iloc[3])*0.3
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[4] = np.array(
        QuantRegAbs_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*0.2 + np.array(ARMA_nn_Forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*0.8

    return combined_model
