import pandas as pd
import numpy as np
import holidays


def get_season_mapping(energy_df):

    energy_df['month'] = energy_df.index.month

    # keep summer as a base
    season_mapping = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'autumn': [9, 10, 11]
    }

    # suppress copy warning
    pd.options.mode.chained_assignment = None

    # create season dummy variable
    for season, months in season_mapping.items():
        energy_df[season] = energy_df['month'].apply(
            lambda x: 1 if x in months else 0)

    energy_df = energy_df.drop(columns=['month'])

    return (energy_df)


def get_day_mapping(energy_df):

    energy_df['weekday'] = energy_df.index.weekday

    # keep sunday as a base
    day_mapping = {
        'weekday': [0, 1, 2, 3, 4],
        'saturday': [5]
    }

    # create day dummy variable
    for day, weekday in day_mapping.items():
        energy_df[day] = energy_df['weekday'].apply(
            lambda x: 1 if x in weekday else 0)

    energy_df = energy_df.drop(columns=['weekday'])

    return (energy_df)


def get_hour_mapping(data_df):

    data_df.loc[:, 'hour'] = data_df.index.hour

    data_df = pd.get_dummies(
        data_df, columns=['hour'], prefix=['hour'], dtype=int, drop_first=True)

    return (data_df)


def get_time_mapping(energy_df):

    energy_df['hour'] = energy_df.index.hour

    # keep very low consumption as base
    hour_mapping = {
        'lc': [6, 22, 23],
        'mc': [7, 20, 21],
        'hc': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    }

    # Create dummy variables for each consumption type
    for cons, hours in hour_mapping.items():
        energy_df[cons] = energy_df['hour'].apply(
            lambda x: 1 if x in hours else 0)

    # Drop the 'hour' column
    energy_df = energy_df.drop(columns=['hour'])

    return energy_df


def get_holiday_mapping(energy_df):

    energy_df['time'] = energy_df.index
    holidays_de = holidays.DE()

    energy_df['holiday'] = energy_df['time'].apply(
        lambda x: 1 if x in holidays_de else 0)

    # Drop the 'time' column
    energy_df = energy_df.drop(columns=['time'])

    return (energy_df)
