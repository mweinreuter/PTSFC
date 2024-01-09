# finished

import numpy as np
import pandas as pd

import ephem
import datetime

# calculate daily sun hours


def calculate_sun_hours(energydata):

    start_date = energydata.index.min()
    end_date = energydata.index.max()

    # central point in Germany
    latitude = 50.1109
    longitude = 8.6821

    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)

    sun = ephem.Sun()

    date_format = "%Y-%m-%d %H:%M:%S"
    current_date = start_date

    # Create an empty DataFrame with columns
    sun_data = []

    while current_date <= end_date:
        observer.date = current_date.strftime(date_format)
        sunrise = ephem.localtime(observer.next_rising(sun))
        sunset = ephem.localtime(observer.next_setting(sun))

        # Calculate sun hours and append to the list
        sun_hours = (sunset - sunrise).seconds/(60*60)
        sun_data.append({'date': current_date, 'sun_hours': sun_hours})

        # Move to the next day
        current_date += datetime.timedelta(days=1)

    sun_df = pd.DataFrame(sun_data)

    return sun_df


def ec_sun_hours_merge(energydata, sun_df=pd.DataFrame):

    if sun_df.empty:
        sun_df = calculate_sun_hours(energydata)

    energydata['date'] = pd.to_datetime(energydata.index.date)
    energydata = energydata.reset_index()

    # merge data
    energy_merged = pd.merge(energydata, sun_df, how='left', on='date').set_index(
        'date_time').drop(columns={'date'})

    return (energy_merged)
