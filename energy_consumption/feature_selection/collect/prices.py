import pandas as pd
import numpy as np

# monthly energy prices --> one lag necesary due to insufficient data


def get_energy_prices():
    energyprices = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/energy_consumption/feature_selection/data/energy_prices.csv')

    energyprices['date'] = energyprices['date_time'].str.split(';').str[0]
    energyprices['date'] = pd.to_datetime(
        energyprices['date'], format='%d.%m.%Y').dt.date

    # prepare for merge
    energyprices['year'] = pd.to_datetime(energyprices['date']).dt.year
    energyprices['month'] = pd.to_datetime(energyprices['date']).dt.month

    energyprices = energyprices.drop(columns=['date_time']).rename(
        columns={'Unnamed: 1': 'price_mean_monthly'})

    # take log --> prices
    energyprices['log_price_pm'] = energyprices['price_mean_monthly'].apply(
        lambda x: np.log1p(x))
    energyprices['log_price_pm_lag1'] = energyprices['log_price_pm'].shift(1)

    return energyprices.drop(columns='price_mean_monthly')


def add_energy_prices(energydata, energyprices=None):
    if energyprices is None:
        energyprices = get_energy_prices()

    energydata = energydata.reset_index()
    energydata['date_time'] = pd.to_datetime(energydata['date_time'])

    # prepare for merge
    energydata['year'] = energydata['date_time'].dt.year
    energydata['month'] = energydata['date_time'].dt.month

    merged = pd.merge(energydata, energyprices,  how='left', left_on=['year', 'month'],
                      right_on=['year', 'month']).set_index('date_time').drop(columns=['year', 'month', 'date'])

    return merged
