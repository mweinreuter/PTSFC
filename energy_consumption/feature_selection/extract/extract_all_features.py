import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from energy_consumption.feature_selection.collect import dummy_mapping, political_instability, weather_sunhours, weather_tempandwind, population, production_index, prices
from energy_consumption.feature_selection.clean.impute_outliers import impute_outliers


def get_energy_and_features(energydata=np.nan, lasso=False, quantReg=False, quantReg_advanced=False, quantReg_final=False, ts=False, only_features=False):

    if type(energydata) == float:
        energydata = pd.read_csv(
            'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\feature_selection\\data\\historical_data.csv')
        energydata['date_time'] = pd.to_datetime(
            energydata['date_time'], format='%Y-%m-%d %H:%M:%S')
        energydata = energydata.set_index("date_time")
        energydata = impute_outliers(energydata)

    if lasso == True:  # try to change
        print('did you update weather and index?')
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings)
                      .pipe(political_instability.ec_dax_merge)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      .pipe(population.get_population)
                      )
        energydata = energydata.drop(
            columns=['close_weekly', 'volatility_weekly']).dropna(subset='abs_log_ret_weekly')

    if quantReg_final == True:
        print('did you update weather?')
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings_advanced)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      )
        energydata = energydata.drop(columns=['wspd'])

    if quantReg == True:
        print('did you update weather and index?')
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      )

    if quantReg_advanced == True:
        print('did you update weather and index?')
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings_advanced)
                      .pipe(political_instability.ec_dax_merge)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      .pipe(population.get_population)
                      )
        energydata = energydata.drop(
            columns=['close_weekly', 'volatility_weekly']).dropna(subset='abs_log_ret_weekly')

    if ts == True:
        energydata = dummy_mapping.get_mappings_advanced(energydata)

    if only_features == True:
        energydata = (energydata
                      .pipe(political_instability.ec_dax_merge)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      .pipe(prices.add_energy_prices)
                      .pipe(population.get_population)
                      )

    return energydata


def get_energy_and_standardized_features(energydata=np.nan, lasso=False, knn=False):

    if type(energydata) == float:
        energydata = pd.read_csv(
            'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\feature_selection\\data\\historical_data.csv')
        energydata['date_time'] = pd.to_datetime(
            energydata['date_time'], format='%Y-%m-%d %H:%M:%S')
        energydata = energydata.set_index("date_time")
        energydata = impute_outliers(energydata)

    if lasso == True:
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings)
                      .pipe(political_instability.ec_dax_merge)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      .pipe(population.get_population)
                      )
        energydata = energydata.drop(
            columns=['close_weekly', 'volatility_weekly']).dropna(subset='abs_log_ret_weekly')

    if knn == True:
        energydata = (energydata
                      .pipe(dummy_mapping.get_mappings_knn)
                      .pipe(weather_sunhours.ec_sun_hours_merge)
                      .pipe(weather_tempandwind.ec_weather_merge)
                      .pipe(production_index.merge_production_indexes)[0]
                      .pipe(population.get_population)
                      )

    scaler = StandardScaler()
    # check if energydata only contains predictors or target as well
    if 'energy_consumption' in energydata.columns:
        X = energydata.drop(columns=['energy_consumption'])
        y = energydata.reset_index()[['energy_consumption', 'date_time']]
    else:
        X = energydata

    # Fit the scaler on your data and transform the features
    X_standardized = scaler.fit_transform(X)
    X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)

    if 'energy_consumption' in energydata.columns:
        X_standardized_df['energy_consumption'] = y['energy_consumption']
        X_standardized_df['date_time'] = y['date_time']
    else:
        X_standardized_df['date_time'] = energydata.index

    X_standardized_df = X_standardized_df.set_index('date_time')

    return X_standardized_df
