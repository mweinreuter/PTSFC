import pandas as pd


def get_season_mapping(data_df):

    data_df['month'] = data_df.index.month

    # define the mapping of months to seasons
    season_mapping = {
        'winter': [12, 1, 2],
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11]
    }

    # create season dummy variable
    for season, months in season_mapping.items():
        data_df[season] = data_df['month'].apply(
            lambda x: 1 if x in months else 0)

    data_df = data_df.drop(columns=['month'])

    return (data_df)


def get_day_mapping(data_df):

    data_df['weekday'] = data_df.index.weekday

    day_mapping = {
        'working_day': list(range(4)),
        'saturday': [5],
        'sunday': [6],
    }

    # create day dummy variable
    for day, weekday in day_mapping.items():
        data_df[day] = data_df['weekday'].apply(
            lambda x: 1 if x in weekday else 0)

    data_df = data_df.drop(columns=['weekday'])

    return (data_df)


def get_hour_mapping(data_df):

    data_df['hour'] = data_df.index.hour

    data_df = pd.get_dummies(
        data_df, columns=['hour'], prefix=['hour'], dtype=int)

    return (data_df)
