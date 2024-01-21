# not finished --> only historical data

import pandas as pd
import numpy as np

# Wind und Temperatur

# Erfurt as middle of Germany
# source: https://meteostat.net/de/place/de/erfurt?s=10554&t=2016-11-28/2024-01-15


def get_weather_data_erfurt():
    weather = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/energy_consumption/feature_selection/data/weather_01172024.csv')
    weather['date'] = pd.to_datetime(
        weather['date'], format='%m/%d/%Y')
    return weather[['date', 'tavg', 'wspd']]


def ec_weather_merge(energydata, weather=pd.DataFrame):
    if weather.empty:
        weather = get_weather_data_erfurt()
    energydata['date'] = pd.to_datetime(energydata.index.date)
    energydata = energydata.reset_index()

    # merge data
    energy_merged = pd.merge(energydata, weather, how='left', on='date').set_index(
        'date_time').drop(columns={'date'})

    return energy_merged
