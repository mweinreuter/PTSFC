import pandas as pd
import numpy as np
import holidays

from datetime import date


def get_mappings(energy_df):
    return (
        energy_df
        .pipe(get_hour_mapping)
        .pipe(get_day_mapping)
        .pipe(get_month_mapping)
        .pipe(get_year_mapping)
        .pipe(get_holiday_mapping_advanced)
    )


def get_mappings_fs(energy_df):
    return (
        energy_df
        .pipe(get_kmeans_hour_mapping)
        .pipe(get_workday_mapping)
        .pipe(get_season_mapping)
        .pipe(get_holiday_mapping_advanced)
    )


def get_mappings_fs_compare(energy_df):
    return (
        energy_df
        .pipe(get_hour_mapping)
        .pipe(get_day_mapping)
        .pipe(get_season_mapping)
        .pipe(get_holiday_mapping_advanced)
    )


def get_mappings_advanced(energy_df):
    return (
        energy_df
        .pipe(get_season_mapping)
        .pipe(get_day_mapping)
        .pipe(get_hour_mapping)
        .pipe(get_year_mapping)
        .pipe(get_holiday_mapping_advanced)
    )


def get_mappings_baseline(energy_df):
    return (
        energy_df
        .pipe(get_hour_mapping)
        .pipe(get_month_mapping)
    )


def get_season_mapping(energy_df):

    energy_df['month'] = energy_df.index.month

    # keep summer as a base
    # merge spring and autumn ~ transition season
    season_mapping = {
        'winter': [12, 1, 2],
        'spring_autumn': [3, 4, 5, 9, 10, 11]
    }

    # suppress copy warning
    pd.options.mode.chained_assignment = None

    # create season dummy variable
    for season, months in season_mapping.items():
        energy_df[season] = energy_df['month'].apply(
            lambda x: 1 if x in months else 0)

    energy_df = energy_df.drop(columns=['month'])

    return (energy_df)


def get_workday_mapping(energy_df):

    energy_df['weekday'] = energy_df.index.weekday
    energy_df['saturday'] = energy_df['weekday'].apply(
        lambda x: 1 if x in [5] else 0)
    energy_df['working_day'] = energy_df['weekday'].apply(
        lambda x: 1 if x in [0, 1, 2, 3, 4] else 0)
    energy_df = energy_df.drop(columns=['weekday'])

    return energy_df


def get_day_mapping(energy_df):

    days = list(range(1, 7))  # leave out monday
    energy_df.loc[:, 'day'] = energy_df.index.weekday
    for d in days:
        name = f'day_{d}'
        energy_df[name] = np.where(energy_df['day'] == d, 1, 0)

    return energy_df.drop(columns=['day'])


def get_hour_mapping(energy_df):

    hours = list(range(1, 24))  # leave out 0:00 am
    energy_df.loc[:, 'hour'] = energy_df.index.hour
    for h in hours:
        name = f'hour_{h}'
        energy_df[name] = np.where(energy_df['hour'] == h, 1, 0)

    return energy_df.drop(columns=['hour'])


def get_month_mapping(energy_df):

    months = list(range(2, 13))  # leave out january
    energy_df.loc[:, 'month'] = energy_df.index.month
    for m in months:
        name = f'month_{m}'
        energy_df[name] = np.where(energy_df['month'] == m, 1, 0)

    return energy_df.drop(columns=['month'])


def get_year_mapping(energy_df):

    # need to drop reference variable in the end
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    energy_df.loc[:, 'year'] = energy_df.index.year
    for y in years:
        name = f'year_{y}'
        energy_df[name] = np.where(energy_df['year'] == y, 1, 0)

    return energy_df.drop(columns='year')


def get_kmeans_hour_mapping(energy_df):

    energy_df['hour'] = energy_df.index.hour

    # keep very low consumption as base --> 'period0': [0,5,23]
    # after feature engineering: merged period 7 (orignially, containing solely 22)
    # and period 4 (containing 6 and 21)
    hour_mapping = {
        'period1': [13],
        'period2': [1, 2, 3, 4],
        'period3': [8, 14, 15, 16, 17, 18, 19],
        'period4': [6, 21, 22],
        'period5': [7, 20],
        'period6': [10, 11, 12]
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


def get_holiday_mapping_advanced(energy_df):
    """includes bridge days"""

    energy_df['timestamp'] = energy_df.index

    # Create a custom holidays object that includes 31-12 as a holiday
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    holidays_de = holidays.Germany(years=years)
    for y in years:
        holidays_de.append({date(y, 12, 31): "New Year's Eve"})

    # Mark actual holidays
    energy_df['holiday'] = energy_df['timestamp'].apply(
        lambda x: 1 if x in holidays_de else 0)

    # Identify bridge days (Monday)
    energy_df['bridge_day_monday'] = energy_df['timestamp'].apply(
        lambda x: 1 if x.weekday() == 0 and (x + pd.DateOffset(days=1)) in holidays_de else 0)

    # Identify bridge days (Friday)
    energy_df['bridge_day_friday'] = energy_df['timestamp'].apply(
        lambda x: 1 if x.weekday() == 4 and (x + pd.DateOffset(days=-1)) in holidays_de else 0)

    # Combine actual holidays and bridge days
    energy_df['holiday'] = energy_df['holiday'] | energy_df['bridge_day_monday'] | energy_df['bridge_day_friday']

    energy_df = energy_df.drop(
        columns=['timestamp', 'bridge_day_monday', 'bridge_day_friday'])
    return energy_df
