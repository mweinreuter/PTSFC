import pandas as pd
import numpy as np

import requests
from datetime import datetime, timedelta
from tqdm import tqdm


def get_data():  # to do: fasten

    # get all available time stamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_quarterhour.json"
    response = requests.get(stampsurl)
    # ignore first 4 years (don't need those in the baseline and speeds the code up a bit)
    timestamps = list(response.json()["timestamps"])[
        7*52:]

    col_names = ['date_time', 'energy_consumption']
    energydata = pd.DataFrame(columns=col_names)

    # loop over all available timestamps
    for stamp in tqdm(timestamps):

        dataurl = "https://www.smard.de/app/chart_data/410/DE/410_DE_quarterhour_" + \
            str(stamp) + ".json"
        response = requests.get(dataurl)
        rawdata = response.json()["series"]

        for i in range(len(rawdata)):

            rawdata[i][0] = datetime.fromtimestamp(
                int(str(rawdata[i][0])[:10])).strftime("%Y-%m-%d %H:%M:%S")

        energydata = pd.concat(
            [energydata, pd.DataFrame(rawdata, columns=col_names)])

    energydata = energydata.dropna()
    energydata["date_time"] = pd.to_datetime(energydata.date_time)

    # set date_time as index
    energydata.set_index("date_time", inplace=True)

    # resample
    energydata = energydata.resample("1h", label="left").sum()

    # transform MWh in GWh
    energydata['energy_consumption'] = energydata['energy_consumption'].astype(
        float)/1000

    return set_last_hour(energydata)


def set_last_hour(energydata):

    # Find the last timestamp in the DataFrame
    last_timestamp = energydata.index[-1]
    last_date = str(last_timestamp.date())

    # Always determine the last observation time to be at 12am
    last_observation_time = '12:00:00'

    # Find index of the last observation
    last_observation_datetime = pd.to_datetime(
        last_date + ' ' + last_observation_time)

    if last_observation_datetime in energydata.index:
        last_observation_index = energydata.index.get_loc(
            last_observation_datetime)
    else:
        last_observation_datetime = pd.to_datetime(
            last_observation_datetime) - timedelta(days=1)
        last_observation_index = energydata.index.get_loc(
            last_observation_datetime)

    energydata = energydata.iloc[:last_observation_index + 1]

    return (energydata)
