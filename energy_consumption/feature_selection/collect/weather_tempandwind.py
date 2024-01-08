# not finished --> only historical data

import pandas as pd
import numpy as np

# Wind und Temperatur

# Erfurt as middle of Germany


def get_hist_weather_data_erfurt():
    historicalweather = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/energy_consumption/feature_selection/data/historical_weather_data.csv')
    historicalweather['date'] = pd.to_datetime(
        historicalweather['date'], format='%m/%d/%Y')
    return historicalweather[['date', 'tavg', 'wspd']]

# need to write function to get weather forecasts


def get_weather_forecasts():
    weatherforecasts = pd.DataFrame()
    return weatherforecasts


def ec_weather_merge(energydata, weather=pd.DataFrame, train=True):

    if weather.empty and train == True:
        weather = get_hist_weather_data_erfurt()

    elif weather.empty and train == False:
        weather = get_weather_forecasts()

    energydata['date'] = pd.to_datetime(energydata.index.date)
    energydata = energydata.reset_index()

    # merge data
    energy_merged = pd.merge(weather, energydata, how='left', on='date').set_index(
        'date_time').drop(columns={'date'}).dropna(subset=['energy_consumption'])

    return energy_merged
