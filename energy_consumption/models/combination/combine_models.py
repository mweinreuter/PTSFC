import numpy as np
import pandas as pd

from energy_consumption.help_functions.get_energy_data import get_data


def get_combined_models(models, energydata=pd.DataFrame, wednesday_morning=False):
    """
    list of models, each model 
    contains dataframe of forecasting results, 
    list of boolean values of each horizon 
    weights, if boolean value is true 
    """

    if energydata.empty:
        energydata = get_data(wednesday_morning=wednesday_morning)

    # calculate forecasts
    forecasts = {}
    for model in models:
        name = str('forecast_model_' + model['name'])
        forecasts[name] = model["function"](energydata)

    # create combined data frame
    combined_model = list(forecasts.values())[0].copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = 0

    # for each model and horizon: add weighted forecasts, if horizon_integration = True
    for model in models:
        name = str('forecast_model_' + model['name'])
        for h in range(6):
            if model['horizon_integration'][h] == True:
                to_add = model['weights'][h] * np.array(
                    forecasts[name].iloc[h].loc['q0.025':'q0.975'])
                combined_model.loc[combined_model.index[h],
                                   'q0.025':'q0.975'] += to_add

    return (combined_model)
