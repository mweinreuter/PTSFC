import pandas as pd
import numpy as np

from scipy.stats import norm
from math import sqrt

correlated_pairs = [(1, 2), (1, 12), (1, 13), (1, 14),
                    (1, 15), (1, 16), (1, 17), (2, 12),
                    (2, 13), (2, 14), (2, 16), (3, 4),
                    (5, 7), (6, 7), (6, 8), (6, 9), (6, 10),
                    (7, 8), (7, 9), (7, 10), (8, 9), (8, 10),
                    (9, 10), (12, 16), (12, 17), (13, 14),
                    (13, 15), (13, 16), (14, 15), (15, 16),
                    (15, 17), (16, 17)]

column_set = {'winter_spring_autumn', 'saturday_working_day',
              'period1_period3', 'period2_period3',
              'period2_period4', 'period2_period5', 'period2_period6',
              'period3_period4', 'period3_period5',
              'period3_period6', 'period4_period5',
              'period4_period6', 'period5_period6'}


def get_interaction_and_pol_terms(X):
    ''' X needs to be in lasso shape (18 columns)'''

    if len(X.columns) == 18:

        # Create copy of the original predictor matrix
        X_int_pol = X.copy()

        # Add interaction terms for selected pairs
        for pair in correlated_pairs:
            col1, col2 = X.columns[pair[0]], X.columns[pair[1]]
            name = f"{col1}_{col2}"
            X_int_pol[name] = X_int_pol[col1] * \
                X_int_pol[col2]

        # Drop interaction terms for dummies of same category
        columns_to_drop = column_set.intersection(set(X_int_pol.columns))
        X_int_pol = X_int_pol.drop(columns=list(columns_to_drop))

        # Add polynomials
        X_int_pol['tavg_2'] = X_int_pol['tavg']**2
        X_int_pol['wspd_2'] = X_int_pol['wspd']**2
        X_int_pol['sun_hours_2'] = X_int_pol['sun_hours']**2

        return X_int_pol

    else:
        print('X has wrong number of columns')
        return X


def estimate_forecast_std(model_variance, horizon):
    return sqrt(model_variance)*sqrt(round(horizon/24, 1))


def get_quantiles(mean_est, std_est, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):

    column_names = [f'q{q}' for q in quantiles]
    quantile_df = pd.DataFrame(np.nan, index=range(
        len(mean_est)), columns=column_names)

    # input two np.arrays
    for i in range(len(mean_est)):
        quantile_df.loc[i] = np.array(
            mean_est[i] + std_est[i]*norm.ppf(quantiles, loc=0))

    return quantile_df
