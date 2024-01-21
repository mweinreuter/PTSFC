import pandas as pd
import numpy as np
import statsmodels.api as sm

from dax.help_functions.get_dax_data import get_prepared_data
from evaluation.help_functions.prepare_data import next_working_days
from dax.feature_selection.variables.get_unemployment_rate import merge_unemployment_rate
from dax.feature_selection.variables.get_us_volatility import get_us_volatility_data


def get_QuantRegFeatures_forecasts(daxdata=pd.DataFrame(), quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):

    if daxdata.empty:
        daxdata = get_prepared_data()

    # add unemployment data
    daxdata = merge_unemployment_rate(daxdata)[0]
    daxdata = get_us_volatility_data(daxdata)

    column_names = [f'q{q}' for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    start_date_dates = max(daxdata.index).strftime('%Y-%m-%d')
    dates = next_working_days(start_date_dates, 5)
    quantile_df = pd.DataFrame(index=dates, columns=column_names)

    for h in range(1, 6):

        name = f'LogRetLag{h}'
        X = pd.DataFrame(
            daxdata.iloc[:-h][[name, 'rate', 'volatility_weekly']].copy())
        X.insert(0, column='intercept', value=np.ones(shape=(len(daxdata)-h)))

        Y = daxdata[[name]].shift(-h).iloc[:-h].copy()
        Y = Y.rename(columns={name: f"lr{h}daysahead"})
        model_qr_temp = sm.QuantReg(endog=Y, exog=X)

        R_t = pd.DataFrame(
            daxdata.iloc[-1:][[name, 'rate', 'volatility_weekly']].copy())
        R_t.insert(0, column='intercept', value=1)

        for q in quantiles:
            model_quantile = model_qr_temp.fit(q=q)
            # Calculate forecasts for R_t using the fitted model for the current quantile
            forecast_temp = model_quantile.predict(R_t)
            quantile_df.loc[dates[h-1]][f'q{q}'] = forecast_temp[0]

    date_st = daxdata.index[-1].strftime('%Y-%m-%d')
    quantile_df.insert(0, 'forecast_date', date_st)
    quantile_df.insert(1, 'target', 'DAX')
    quantile_df.insert(
        2, "horizon", [str(i) + " day" for i in (1, 2, 5, 6, 7)])
    quantile_df.index.name = "date_time"

    return quantile_df
