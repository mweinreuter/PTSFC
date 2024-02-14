import numpy as np
import math
from scipy.stats import norm, t


def get_norm_quantiles(variance, quantiles):
    return 0 + math.sqrt(variance)*norm.ppf(quantiles, loc=0)


def get_norm_quantiles_mean(pair, quantiles):
    mean, variance = pair
    variance = np.array(variance)
    return mean + math.sqrt(variance)*norm.ppf(quantiles, loc=0)


def get_t_quantiles(tuple, quantiles):
    t_df, variance, mean_est = tuple
    return mean_est + math.sqrt(variance) * t.ppf(quantiles, df=t_df)
