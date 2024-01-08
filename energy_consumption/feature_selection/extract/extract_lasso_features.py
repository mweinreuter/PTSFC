import pandas as pd

from sklearn.preprocessing import StandardScaler

from energy_consumption.feature_selection.collect import dummy_mapping, political_instability, weather_sunhours, weather_tempandwind, population, production_index, prices


def get_energy_and_features(energydata=pd.DataFrame):

    if energydata.empty:
        energydata = pd.read_csv(
            'c:\\Users\\Maria\\Documents\\Studium\\Pyhton Projekte\\PTSFC\\energy_consumption\\feature_selection\\data\\historical_data.csv')
        energydata['date_time'] = pd.to_datetime(
            energydata['date_time'], format='%Y-%m-%d %H:%M:%S')
        energydata = energydata.set_index("date_time")
    else:
        print('Updated wind, temp, production_index?')

    energydata = (energydata
                  .pipe(dummy_mapping.get_mappings)
                  .pipe(political_instability.ec_dax_merge)
                  .pipe(weather_sunhours.ec_sun_hours_merge)
                  # .pipe(weather_tempandwind.ec_weather_merge)
                  .pipe(production_index.merge_production_indexes)
                  .pipe(population.get_population)
                  )

    return energydata


def get_energy_and_standardized_features(train=True):

    if train == True:
        energydata = get_energy_and_features(train)

    scaler = StandardScaler()

    # Assuming X is your feature matrix (excluding the target variable 'energy_consumption')
    X = energydata.drop(columns=['energy_consumption'])
    y = energydata.reset_index()['energy_consumption']

    # Fit the scaler on your data and transform the features
    X_standardized = scaler.fit_transform(X)

    X_standardized_df = pd.DataFrame(X_standardized, columns=X.columns)
    X_standardized_df['energy_consumption'] = y

    return X_standardized_df
