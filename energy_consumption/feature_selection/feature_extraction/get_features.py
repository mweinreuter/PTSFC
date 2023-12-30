import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline

from energy_consumption.feature_selection.feature_extraction import dummy_mapping, political_instability, weather_sunhours, weather_tempandwind, production_index, prices


def get_energy_and_features(train=True):

    if train == True:
        energydata = pd.read_csv(
            'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\historical_data.csv')
        energydata['date_time'] = pd.to_datetime(
            energydata['date_time'], format='%Y-%m-%d %H:%M:%S')
        energydata = energydata.set_index("date_time")
    else:
        print('Updated wind, temp, production_index and price?')

    return (
        energydata
        .pipe(dummy_mapping.get_mappings)
        .pipe(political_instability.ec_dax_merge)
        .pipe(weather_sunhours.ec_sun_hours_merge)
        .pipe(weather_tempandwind.ec_weather_merge)
        .pipe(production_index.merge_production_indexes)
        .pipe(prices.add_energy_prices)
    )
