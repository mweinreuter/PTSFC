import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm

residuals = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\residuals.csv')
distances = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\distances.csv')


def get_quantiles(mean_est, neighbor_distances, quantiles):

    quantile_df = pd.DataFrame()
    quantiles = (np.array(quantiles))

    mean_distances = np.array(distances['mean_distance'])
    distance_specific = neighbor_distances.mean(axis=1)
    distance_ratio = (distance_specific/mean_distances)  # durch?

    mean_residuals = np.array(residuals.mean(axis=0))
    mean_corr = np.array(mean_est) - mean_residuals

    std_dev_residuals = np.array(residuals.std(axis=0))
    for q in quantiles:
        quantile_df[f'q{q}'] = mean_corr+distance_ratio * \
            norm.ppf(q, loc=0)*std_dev_residuals*(0.75)

    return quantile_df
