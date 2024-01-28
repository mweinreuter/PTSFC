import numpy as np
import pandas as pd

from energy_consumption.feature_selection.extract import extract_energy_data

# import required models
from energy_consumption.models.lasso.lasso import get_Lasso_forecasts
from energy_consumption.models.quantreg.quantreg_end import get_QuantReg_forecasts
from energy_consumption.models.quantreg.quantreg_ex import get_QuantRegShort_forecasts
from energy_consumption.models.baseline import get_baseline_forecasts
from energy_consumption.models.knn.knn import get_KNNRegression_forecasts
from old_models.XGBoost import get_XGBoost_forecasts


def combine_selected_models(energydata=pd.DataFrame):

    if type(energydata) == float:
        # use derived optimum for number of years
        energydata = extract_energy_data.get_data(num_years=7)

    QuantReg_forecasts = get_QuantReg_forecasts(energydata)
    QuantRegShort_forecasts = get_QuantRegShort_forecasts(energydata)
    Lasso_forecasts = get_Lasso_forecasts
    XGBoost_forecasts = get_XGBoost_forecasts(energydata)
    KNN_forecasts = get_KNNRegression_forecasts(energydata)
    baseline_forecasts = get_baseline_forecasts(energydata)

    combined_model = QuantReg_forecasts.copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    combined_model = pd.DataFrame(
        index=Lasso_forecasts.index, columns=Lasso_forecasts.columns)
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    combined_model.loc[:, 'q0.025':'q0.975'].iloc[0] = np.array(
        QuantRegShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.7 + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.3
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[1] = np.array(
        Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.3 + np.array(QuantRegShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.7
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[2] = np.array(
        Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.4 + np.array(KNN_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.6
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[3] = np.array(
        QuantRegShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*0.5 + np.array(baseline_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*0.5
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[4] = np.array(
        QuantRegShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*0.6 + np.array(XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*0.4
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[5] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[5])*0.8 + np.array(QuantRegShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[5])*0.2

    combined_model['date_time'] = QuantRegShort_forecasts['date_time']
    combined_model['forecast_date'] = QuantRegShort_forecasts['forecast_date']
    combined_model['target'] = QuantRegShort_forecasts['target']
    combined_model['horizon'] = QuantRegShort_forecasts['horizon']

    return combined_model
