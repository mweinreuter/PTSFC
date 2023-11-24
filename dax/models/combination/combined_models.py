import numpy as np
import pandas as pd
from dax.models.time_series_models.garch11_t import get_garch_11_t
from dax.models.baseline import get_dax_forecasts_baseline

# only combination ob garch(1,1) and baseline possible (in any other case: change import statements)


def combine_models(daxdata=pd.DataFrame, weights=[0.5, 0.5]):

    garch_t_forecasts = get_garch_11_t(daxdata)
    baseline_forecasts = get_dax_forecasts_baseline(daxdata)

    combined_model = garch_t_forecasts.copy()
    combined_model.loc[:, 'q0.025':'q0.975'] = np.nan

    garch_t_forecast_array = np.array(
        garch_t_forecasts.loc[:, 'q0.025':'q0.975'])
    baseline_forecast_array = np.array(
        baseline_forecasts.loc[:, 'q0.025':'q0.975'])

    combined_value_array = weights[0]*garch_t_forecast_array + \
        weights[1]*baseline_forecast_array
    combined_value_array

    for i in range(len(combined_model)):
        combined_model.loc[combined_model.index[i],
                           'q0.025':'q0.975'] = combined_value_array[i]

    return (combined_model)
