import pandas as pd
import numpy as np


def merge_production_indexes(energydata):

    productionindexes = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/energy_consumption/feature_selection/data/production_index.csv')
    productionindexes = productionindexes.drop(columns=['month'])

    # calculate mean of production indexes for each year in each half
    production_means = productionindexes.groupby(
        by=['year', 'half'], as_index=False).mean()
    production_means['year'] = production_means['year'].astype(int)
    production_means['half'] = production_means['half'].astype(int)

    energydata['year'] = energydata.index.year
    energydata['half'] = 0
    energydata['half'][energydata.index.month.isin([7, 8, 9, 10, 11, 12])] = 1
    energydata = energydata.reset_index()

    merged = pd.merge(energydata, production_means,  how='left', left_on=['year', 'half'], right_on=[
                      'year', 'half']).set_index('date_time').drop(columns=['year', 'half'])

    return merged
