import math
from scipy.stats import norm, t


def get_norm_quantiles(variance):
    return 0 + math.sqrt(variance)*norm.ppf([0.025, 0.25, 0.5, 0.75, 0.975], loc=0)


def get_norm_quantiles_mean(pair):
    mean, variance = pair

    # unsure if usage of quantiles of standard normal distribution is correct
    return mean + math.sqrt(variance)*norm.ppf([0.025, 0.25, 0.5, 0.75, 0.975], loc=0)


def get_t_quantiles(pair, mean_est=0):
    t_df, variance = pair
    return mean_est + math.sqrt(variance) * t.ppf([0.025, 0.25, 0.5, 0.75, 0.975], df=t_df)
