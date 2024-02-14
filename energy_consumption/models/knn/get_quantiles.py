import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import norm

residuals = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\data\\residuals_latest_250.csv')
distances = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\data\\distances_latest_250.csv')

residuals_latest = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\data\\residuals_latest.csv')
distances_latest = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\data\\distances_latest.csv')


def get_quantiles(mean_est, neighbor_distances, quantiles):

    quantile_df = pd.DataFrame()
    quantiles = (np.array(quantiles))

    mean_distances = np.array(distances['mean_distance'])
    distance_specific = neighbor_distances.mean(axis=1)
    distance_ratio = (distance_specific/mean_distances)  # durch?

    mean_residuals = np.array(residuals.mean(axis=0))
    mean_corr = np.array(mean_est) - mean_residuals
    # estimate std --> take 1/(N-K), N = 200, K = 54
    std_dev_residuals = np.sqrt(
        np.array((residuals.var(axis=0)) * (200 / (200 - 54))))

    for q in quantiles:
        quantile_df[f'q{q}'] = mean_corr + \
            norm.ppf(q, loc=0)*(distance_ratio *
                                std_dev_residuals + sqrt(1.0417))

    return quantile_df


def get_quantiles_latest(mean_est, neighbor_distances, quantiles):

    quantile_df = pd.DataFrame()
    quantiles = (np.array(quantiles))

    mean_distances = np.array(distances_latest['mean_distance'])
    distance_specific = neighbor_distances.mean(axis=1)
    distance_ratio = (distance_specific/mean_distances)  # durch?

    mean_residuals = np.array(residuals_latest.mean(axis=0))
    mean_corr = np.array(mean_est) - mean_residuals
    # estimate std --> take 1/(N-K), N = 200, K = 54
    std_dev_residuals = np.array(residuals_latest.std(axis=0))

    for q in quantiles:
        quantile_df[f'q{q}'] = mean_corr + \
            norm.ppf(q, loc=0)*(distance_ratio *
                                std_dev_residuals + sqrt(1.0417))

    return quantile_df
