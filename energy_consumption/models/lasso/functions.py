import pandas as pd
import numpy as np

from scipy.stats import norm
from math import sqrt


def get_interaction_and_pol_terms(X):

    # interaction term for sun_hours and tavg
    X['sun_hours_tavg'] = X['sun_hours'] * X['tavg']

    # Add polynomials
    X['tavg_2'] = X['tavg']**2
    X['wspd_2'] = X['wspd']**2
    X['sun_hours_2'] = X['sun_hours']**2

    return X


def get_quantiles(mean_est, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):

    mean_est = np.array(mean_est)
    quantile_df = pd.DataFrame()

    # lasso variance
    residuals = pd.read_csv(
        'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\lasso\\residuals.csv')
    residual_std = np.sqrt(residuals.var(axis=0))

    # add variance due to forecasts of regressors
    std_to_add = sqrt(0.619823324688293)

    for q in quantiles:
        if q < 0.5:
            quantile_df[f'q{q}'] = mean_est + \
                residual_std*norm.ppf(q, loc=0) - std_to_add
        elif q == 0.5:
            quantile_df[f'q{q}'] = mean_est
        else:
            quantile_df[f'q{q}'] = mean_est + \
                residual_std*norm.ppf(q, loc=0) - std_to_add

    return quantile_df
