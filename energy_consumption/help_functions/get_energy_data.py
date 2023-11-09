import pandas as pd


def get_data(file_path):

    energy_data = (pd.read_csv(file_path,
                               delimiter=';', encoding='utf-8')
                   .iloc[:, [0, 1, 3]]
                   .dropna()
                   .rename(columns={'Datum': 'date',
                                    'Anfang': 'beginning',
                                    'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'energy_consumption'}))

    # convert data type and measurement unit of energy consumption (in GWh instead of MWh)
    energy_data = energy_data[~energy_data['energy_consumption'].str.contains(
        '-')]
    energy_data['energy_consumption'] = (energy_data['energy_consumption'].str.replace('.', '', regex=False)
                                         .str.replace(',', '.', regex=False)
                                         .astype(float) / 1000)

    # Set date_time as row index for time series
    energy_data['date'] = pd.to_datetime(
        energy_data['date'], format='%d.%m.%Y')
    energy_data['beginning'] = pd.to_datetime(
        energy_data['beginning'], format='%H:%M').dt.time
    energy_data['date_time'] = energy_data.apply(
        lambda row: pd.Timestamp.combine(row['date'], row['beginning']), axis=1)
    energy_data.set_index('date_time', inplace=True)

    return energy_data
