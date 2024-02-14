import numpy as np
import pandas as pd

from energy_consumption.feature_selection.extract import extract_energy_data

# import required models
from energy_consumption.models.lasso.lasso import get_Lasso_forecasts
from energy_consumption.models.knn.knn import get_KNNRegression_forecasts
from energy_consumption.models.XGBoost.XGBoost import get_XGBoost_forecasts
from energy_consumption.models.quantreg.quantreg_end import get_QuantRegEndShort_forecasts
from energy_consumption.models.quantreg.quantreglags import get_QuantRegExLags_forecasts


def combine_selected_models(energydata=np.nan):

    if type(energydata) == float:
        # use derived optimum for number of years
        energydata = extract_energy_data.get_data()

    QuantRegEndShort_forecasts = get_QuantRegEndShort_forecasts(energydata)
    Lasso_forecasts = get_Lasso_forecasts(energydata)  # corrected
    XGBoost_forecasts = get_XGBoost_forecasts(energydata)
    KNN_forecasts = get_KNNRegression_forecasts(energydata)
    QuantRegExLags = get_QuantRegExLags_forecasts(energydata)

    combined_model = pd.DataFrame(  # corrected
        index=Lasso_forecasts.index, columns=Lasso_forecasts.columns)
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    combined_model.loc[:, 'q0.025':'q0.975'].iloc[0] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.4 + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[0])*0.5 + np.array(QuantRegEndShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[0])*0.1
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[1] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.7 + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[1])*0.2 + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[1])*0.1
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[2] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.6 + np.array(QuantRegEndShort_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.1 + np.array(KNN_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.2 + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[2])*0.05 + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[2])*0.05
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[3] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*(1/4) + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*(3/8) + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[3])*(1/8) + np.array(KNN_forecasts.loc[:, 'q0.025':'q0.975'].iloc[3])*0.25
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[4] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*(1/4) + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*(3/8) + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[4])*(2/8) + + np.array(KNN_forecasts.loc[:, 'q0.025':'q0.975'].iloc[4])*(1/8)
    combined_model.loc[:, 'q0.025':'q0.975'].iloc[5] = np.array(
        XGBoost_forecasts.loc[:, 'q0.025':'q0.975'].iloc[5])*(1/3) + np.array(Lasso_forecasts.loc[:, 'q0.025':'q0.975'].iloc[5])*(1/3) + np.array(QuantRegExLags.loc[:, 'q0.025':'q0.975'].iloc[5])*(1/3)

    combined_model['date_time'] = QuantRegEndShort_forecasts['date_time']
    combined_model['forecast_date'] = QuantRegEndShort_forecasts['forecast_date']
    combined_model['target'] = QuantRegEndShort_forecasts['target']
    combined_model['horizon'] = QuantRegEndShort_forecasts['horizon']

    return combined_model
