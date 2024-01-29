import numpy as np
import pandas as pd

from dax.models.baseline import get_dax_forecasts_baseline
from dax.models.QuantReg.QuantRegVol import get_QuantRegVol_forecasts, get_QuantRegVolShort_forecasts
from dax.models.ARMA_NN.arma_nn import get_arma_nn_forecasts
from dax.models.ARMA_GARCH.arma_garch import get_arma_garch_forecasts


def combine_models(daxdf=pd.DataFrame):

    daxdata0 = daxdf.copy()
    baseline = get_dax_forecasts_baseline(daxdata0)
    daxdata1 = daxdf.copy()
    quantreg = get_QuantRegVol_forecasts(daxdata1)
    daxdata2 = daxdf.copy()
    armagarch = get_arma_garch_forecasts(daxdata2)

    combined_model = quantreg.copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    combined_model.loc[:, 'q0.025':'q0.975'].iloc[0] = np.array(
        quantreg.loc[:, 'q0.025':'q0.975'].iloc[0])
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[1] = np.array(
        armagarch.loc[:, 'q0.025':'q0.975'].iloc[1])
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[2] = np.array(
        armagarch.loc[:, 'q0.025':'q0.975'].iloc[2])
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[3] = np.array(
        quantreg.loc[:, 'q0.025':'q0.975'].iloc[3])*0.34 + np.array(armagarch.loc[:, 'q0.025':'q0.975'].iloc[3])*0.33 + np.array(baseline.loc[:, 'q0.025':'q0.975'].iloc[3])*0.33
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[4] = np.array(
        quantreg.loc[:, 'q0.025':'q0.975'].iloc[4])*0.5 + np.array(armagarch.loc[:, 'q0.025':'q0.975'].iloc[4])*0.2 + np.array(baseline.loc[:, 'q0.025':'q0.975'].iloc[4])*0.3

    return combined_model
