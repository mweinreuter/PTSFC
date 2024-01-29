import pandas as pd
import numpy as np

import arch
import pmdarima

from dax.models.ARMA_GARCH.get_quantiles import get_t_quantiles
from evaluation.help_functions.prepare_data import next_working_days
from dax.help_functions.get_dax_data import get_prepared_data

pars = {
    1: [1, 1],  # criterion: AIC, BIC
    2: [1, 1],  # criterion: AIC, BIC
    3: [1, 1],  # criterion: BIC
    4: [1, 2],  # criterion: BIC
    5: [1, 1]   # criterion: BIC
}


def get_arma_garch_forecasts(daxdata=pd.DataFrame, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], submission=True):

    if daxdata.empty:
        daxdata = get_prepared_data()
    date_st = daxdata.index[-1].strftime('%Y-%m-%d')

    horizon_estimates = {}
    for h in [1, 2, 3, 4, 5]:

        temp_model = pmdarima.auto_arima(
            daxdata[f'LogRetLag{h}'], suppress_warnings=True)
        temp_residuals = temp_model.arima_res_.resid

        model = arch.arch_model(
            temp_residuals, vol='Garch', p=pars[h][0], q=pars[h][1], dist='t')
        result = model.fit(disp='off')

        # predict mean, variance and df
        garch_forecast = result.forecast(horizon=h)
        horizon_name = f'h.{h}'
        variance_prediction = garch_forecast.variance[horizon_name].iloc[-1]
        mean_prediction = garch_forecast.mean[horizon_name].iloc[-1]
        df = int(result.params['nu'])

        # store data
        horizon_estimates[h] = (df, variance_prediction, mean_prediction)
        # get quantiles
    quantiles_est = [get_t_quantiles(tuple, quantiles)
                     for tuple in horizon_estimates.values()]

    # create quantile frame
    column_names = [f'q{q}' for q in quantiles]
    dates = next_working_days(max(daxdata.index), 5)
    quantile_df = pd.DataFrame(quantiles_est, columns=column_names)
    quantile_df['date_time'] = dates
    quantile_df.set_index('date_time', inplace=True)

    # create submission frame
    if submission == True:
        quantile_df.insert(0, 'forecast_date', date_st)
        quantile_df.insert(1, 'target', 'DAX')
        quantile_df.insert(
            2, "horizon", [str(i) + " day" for i in (1, 2, 5, 6, 7)])

    return quantile_df
