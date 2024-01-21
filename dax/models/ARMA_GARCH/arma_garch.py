import pandas as pd
import numpy as np

import arch

from dax.models.ARMA_GARCH.get_quantiles import get_t_quantiles
from evaluation.help_functions.prepare_data import next_working_days
from dax.help_functions.get_dax_data import get_prepared_data

# decided for BIC since models have less parameters (simplicity)
par_opt_zero_mean = {
    1: [2, 1],  # criterion: BIC
    2: [1, 4],  # criterion: AIC, BIC
    3: [1, 4],  # criterion: AIC, BIC
    4: [1, 4],  # criterion: AIC, BIC
    5: [1, 4],  # criterion: AIC, BIC
}

par_opt_ar_mean = {
    1: [1, 1, 4],  # criterion: AIC, BIC
    2: [1, 1, 4],  # criterion: AIC, BIC
    3: [1, 1, 4],  # criterion: BIC
    4: [2, 1, 4],  # criterion: BIC
    5: [1, 1, 4]   # criterion: BIC
}


def get_arma_garch11_forecasts(daxdata=pd.DataFrame, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], basic=True, opt_pq=False, opt_lag_pq=False, submission=True):
    ''' basic --> GARCH(1,1) with zero mean'''

    if daxdata.empty:
        daxdata = get_prepared_data()
    date_st = daxdata.index[-1].strftime('%Y-%m-%d')

    horizon_estimates = {}
    for h in [1, 2, 3, 4, 5]:

        if basic == True:
            model = arch.arch_model(
                daxdata.iloc[:, h], mean='zero', vol='Garch', p=1, q=1, dist='t')
            result = model.fit(disp='off')

        elif opt_pq == True:
            p, q = par_opt_zero_mean[h][0], par_opt_zero_mean[h][1]
            model = arch.arch_model(
                daxdata.iloc[:, h], mean='zero', vol='Garch', p=p, q=q, dist='t')
            result = model.fit(disp='off')

        elif opt_lag_pq == True:
            p, q, lag = par_opt_ar_mean[h][0], par_opt_ar_mean[h][1], par_opt_ar_mean[h][2]
            model = arch.arch_model(
                daxdata.iloc[:, h], mean='AR', lags=lag, vol='Garch', p=p, q=q, dist='t')
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
    quantiles_est = [get_t_quantiles(tuple)
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


def get_arma_garch_opt_pq_forecasts(daxdata=pd.DataFrame, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], submission=True):
    ''' opt_pq --> GARCH (p,q) with zero mean and opt. p,q values '''

    if daxdata.empty:
        daxdata = get_prepared_data()
    date_st = daxdata.index[-1].strftime('%Y-%m-%d')

    horizon_estimates = {}
    for h in [1, 2, 3, 4, 5]:

        p, q = par_opt_zero_mean[h][0], par_opt_zero_mean[h][1]
        model = arch.arch_model(
            daxdata.iloc[:, h], mean='zero', vol='Garch', p=p, q=q, dist='t')
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
    quantiles_est = [get_t_quantiles(tuple)
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


def get_arma_garch_opt_pq_lag_forecasts(daxdata=pd.DataFrame, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], submission=True):
    ''' opt_lag_pq --> AR(lag)-GARCH (p,q) model with opt. values '''

    if daxdata.empty:
        daxdata = get_prepared_data()
    date_st = daxdata.index[-1].strftime('%Y-%m-%d')

    horizon_estimates = {}
    for h in [1, 2, 3, 4, 5]:

        p, q, lag = par_opt_ar_mean[h][0], par_opt_ar_mean[h][1], par_opt_ar_mean[h][2]
        model = arch.arch_model(
            daxdata.iloc[:, h], mean='AR', lags=lag, vol='Garch', p=p, q=q, dist='t')
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
    quantiles_est = [get_t_quantiles(tuple)
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
