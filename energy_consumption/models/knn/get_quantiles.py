import pandas as pd
import numpy as np
from math import sqrt

residuals = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\residuals.csv')
distances = pd.read_csv(
    'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\models\\knn\\distances.csv')


def get_quantiles(mean_est, neighbor_distances, indexes, quantiles):

    column_names = [f'q{q}' for q in quantiles]
    quantile_df = pd.DataFrame(columns=column_names)
    quantiles = list(100*np.array(quantiles))

    # input two np.arrays
    for i in indexes:

        # weighten percentiles by distance ration
        mean_distance = np.mean(neighbor_distances[i])
        distance_ratio = mean_distance/distances.iloc[i, 1]

        # estimate quantile
        name = f'index_{i}'
        print(np.array(mean_est[i] + distance_ratio *
              np.percentile(residuals[name], quantiles)))
        quantile_df.loc[i] = np.array(
            mean_est[i] + distance_ratio*np.percentile(residuals[name], quantiles))

    return quantile_df
