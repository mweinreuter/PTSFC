import pandas as pd
import numpy as np


def impute_outliers(energydata):

    # dates for worst outliers
    dates = ['2019-03-31 02:00:00', '2020-03-28 02:00:00',
             '2020-03-29 02:00:00', '2021-10-31 02:00:00',
             '2019-10-27 02:00:00', '2020-10-25 02:00:00']

    for date in dates:
        if date in energydata.index:
            energydata.loc[date, 'energy_consumption'] = np.nan

    # impute with previous observation
    energydata['energy_consumption'] = energydata['energy_consumption'].fillna(
        method='ffill')

    return energydata
