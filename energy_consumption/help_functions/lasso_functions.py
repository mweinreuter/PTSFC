import pandas as pd
import numpy as np

from scipy.stats import norm
from math import sqrt


def estimate_forecast_std(model_variance, horizon):
    return sqrt(model_variance)*sqrt(round(horizon/24, 1))


def get_quantiles(mean_est, std_est, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):

    column_names = [f'q{q}' for q in quantiles]
    quantile_df = pd.DataFrame(np.nan, index=range(6), columns=column_names)

    # input two np.arrays
    for i in range(6):
        quantile_df.loc[i] = np.array(
            mean_est[i] + std_est[i]*norm.ppf(quantiles, loc=0))
    return quantile_df
