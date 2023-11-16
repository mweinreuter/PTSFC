import pandas as pd
import numpy as np
from datetime import datetime

from scipy.stats import norm
from arch import arch_model

from dax.help_functions.get_dax_data import get_data
from dax.help_functions.calculate_returns import calculate_returns


def get_dax_forecasts_basicGARCH():

    # import required data
    daxdata = get_data().iloc[6000:,]
    daxdata = calculate_returns(daxdata, 1)
    daxdata = daxdata.loc[:, ['Close', 'CloseLag1',
                          'RetLag1', 'LogRetLag1']].set_index(daxdata.index.date)

    # GARCH(1,1) with mean zero and normal distribution
    model_garch_zero = arch_model(
        daxdata['LogRetLag1'], mean='zero', vol='GARCH', p=1, q=1).fit()

    # forecast variance
    forecast = model_garch_zero.forecast(horizon=5, reindex=False)
    forecast_variances = forecast.variance

    # define  quantiles
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    # create DataFrame storing quantile forecasts in submission form
    quantile_forecasts = pd.DataFrame({
        "target": "DAX",
        "horizon": [str(i) + " day" for i in (1, 2, 5, 6, 7)],
        "variance": forecast_variances.iloc[0].values})

    # Calculate quantiles for each variance
    for q in quantiles:
        quantile_forecasts[f'q{q}'] = quantile_forecasts['variance'].apply(
            lambda x: norm.ppf(q, loc=0, scale=np.sqrt(x)))

    # adjust DataFrame to merge format
    last_date = daxdata.index[-1]
    first_index_date = last_date + pd.Timedelta(days=1)
    quantile_forecasts.index = pd.date_range(
        start=first_index_date, periods=5, freq='B')

    date_st = (datetime.today().strftime('%Y-%m-%d'))
    quantile_forecasts.insert(0, 'forecast_date', date_st)
    quantile_forecasts = quantile_forecasts.drop(columns={'variance'})

    return (quantile_forecasts)
