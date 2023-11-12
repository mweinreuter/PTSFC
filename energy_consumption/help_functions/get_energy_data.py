import pandas as pd
import numpy as np

import requests
from datetime import datetime
from tqdm import tqdm


def get_data():

    # get all available time stamps
    stampsurl = "https://www.smard.de/app/chart_data/410/DE/index_quarterhour.json"
    response = requests.get(stampsurl)
    # ignore first 4 years (don't need those in the baseline and speeds the code up a bit)
    timestamps = list(response.json()["timestamps"])[4*52:]

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

    return energydata
