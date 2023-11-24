import pandas as pd
import numpy as np

from arch import arch_model
from datetime import timedelta

from evaluation.help_functions.prepare_data import most_recent_thursday, next_working_days
from dax.help_functions.calculate_returns import calculate_returns
from dax.help_functions.get_quantiles import get_norm_quantiles_mean
from dax.help_functions.get_dax_data import get_data


def get_garch_11_ar(daxdata=pd.DataFrame()):

    if daxdata.empty:
        daxdata = get_data().iloc[8000:,]
        start_date_excl = most_recent_thursday(daxdata) - timedelta(days=1)
        daxdata = daxdata.loc[daxdata.index <= start_date_excl]
        date_st = start_date_excl.strftime('%Y-%m-%d')

        daxdata = calculate_returns(daxdata, 5)
        daxdata.index = daxdata.index.date
        daxdata = daxdata.dropna()
    else:
        date_st = daxdata.index[-1].strftime('%Y-%m-%d')

    quantiles = garch11_ars(daxdata)

    quantiles.insert(0, 'forecast_date', date_st)
    quantiles.insert(1, 'target', 'DAX')
    quantiles.insert(2, "horizon", [str(i) + " day" for i in (1, 2, 5, 6, 7)])

    return (quantiles)


# Runs GARCH(1,1) for each horizon
def garch11_ars(df):

    horizon_estimates = {}

    # Predict variance and mean for each horizon via garch11 and ar mean
    for h in range(1, 6):
        mean, variance = garch11_ar(df[f'LogRetLag{h}'], h)
        horizon_estimates[f'horizon{h}'] = (mean, variance)

    # get quantiles via normal distribution and predicted variances and means
    quantiles = [get_norm_quantiles_mean(pair)
                 for pair in horizon_estimates.values()]

    # create submission frame
    column_names = [f'q{q}' for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    dates = next_working_days(max(df.index), 5)
    quantile_df = pd.DataFrame(quantiles, columns=column_names)
    quantile_df['date_time'] = dates
    quantile_df.set_index('date_time', inplace=True)

    return quantile_df


def garch11_ar(df, horizon):

    model = arch_model(df, mean='AR', p=1, q=1)
    model_fit = model.fit()
    predictions = model_fit.forecast(horizon=horizon)

    # Extract mean and variance
    mean_value = np.array(predictions.mean)[0][-1]
    variance_value = predictions.variance.values[0][-1]

    return mean_value, variance_value
