import pandas as pd
import numpy as np

from arch import arch_model

from evaluation.help_functions.prepare_data import next_working_days
from dax.help_functions.get_quantiles import get_norm_quantiles
from dax.help_functions.get_dax_data import get_prepared_data


def get_garch_11_norm(daxdata=pd.DataFrame()):

    if daxdata.empty:
        daxdata = get_prepared_data()

    date_st = daxdata.index[-1].strftime('%Y-%m-%d')
    quantiles = garch11s_norm(daxdata)

    quantiles.insert(0, 'forecast_date', date_st)
    quantiles.insert(1, 'target', 'DAX')
    quantiles.insert(2, "horizon", [str(i) + " day" for i in (1, 2, 5, 6, 7)])

    return (quantiles)


# Runs GARCH(1,1) for each horizon
def garch11s_norm(df):

    # predict variance for each horizon via garch11
    variances = [garch11_norm(df[f'LogRetLag{h}'], h) for h in range(1, 6)]

    # get quantiles via normal distribution and predicted variances
    quantiles = [get_norm_quantiles(v) for v in variances]

    # create submission frame
    column_names = [f'q{q}' for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    dates = next_working_days(max(df.index), 5)
    quantile_df = pd.DataFrame(quantiles, columns=column_names)
    quantile_df['date_time'] = dates
    quantile_df.set_index('date_time', inplace=True)

    return quantile_df


def garch11_norm(df, horizon):

    model = arch_model(df, mean='zero', p=1, q=1)
    model_fit = model.fit()
    predictions = model_fit.forecast(horizon=horizon)
    return predictions.variance.values[0][-1]
