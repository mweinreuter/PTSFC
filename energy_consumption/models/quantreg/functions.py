import numpy as np
import pandas as pd

from energy_consumption.help_functions import get_forecast_timestamps
from energy_consumption.feature_selection.collect import dummy_mapping
from energy_consumption.feature_selection.collect import weather_sunhours

import numpy as np
import pandas as pd

from energy_consumption.help_functions import get_forecast_timestamps
from energy_consumption.feature_selection.collect import dummy_mapping
from energy_consumption.feature_selection.collect import weather_sunhours


def get_energy_and_forecast_quantreg(energydata):

    energydf = energydata.copy()
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydf.index[-1])
    energyforecast['energy_consumption'] = np.nan
    merged = pd.concat([energydf, energyforecast])

    merged.index = pd.to_datetime(merged.index)
    merged = dummy_mapping.get_day_mapping(merged)
    merged = dummy_mapping.get_hour_mapping(merged)
    merged = weather_sunhours.ec_sun_hours_merge(merged)

    merged['weekly_lag'] = merged['energy_consumption'].shift(168)
    merged['yearly_lag'] = merged['energy_consumption'].shift(8760)
    merged = merged[-1100:]

    merged.insert(loc=0, column='constant', value=1)

    energydf = merged[-1100:-100]
    energyforecast = merged[-100:].drop(columns=['energy_consumption'])

    return energydf, energyforecast


def get_energy_and_forecast_quantreg2(energydata):

    energydf = energydata.copy()
    energyforecast = get_forecast_timestamps.forecast_timestamps(
        energydf.index[-1])
    energyforecast['energy_consumption'] = np.nan
    merged = pd.concat([energydf, energyforecast])
    merged.index = pd.to_datetime(merged.index)

    merged['MeanWeek'] = merged['energy_consumption'].rolling(
        window=168).mean()
    merged['MeanWeekLag'] = merged['MeanWeek'].shift(168)
    merged = merged.drop(columns=['MeanWeek'])

    # drop NaNs and select columns
    # merged = merged.dropna(subset=['MeanWeek', 'MeanWeekLag'])[
    #    ["energy_consumption", "MeanWeekLag"]]

    merged = dummy_mapping.get_day_mapping(merged)
    merged = dummy_mapping.get_hour_mapping(merged)
    merged = weather_sunhours.ec_sun_hours_merge(merged)

    merged['weekly_lag'] = merged['energy_consumption'].shift(168)
    merged['weekly_lag2'] = merged['energy_consumption'].shift(168*2)
    merged['yearly_lag'] = merged['energy_consumption'].shift(8760)
    merged = merged[-1100:]

    merged.insert(loc=0, column='constant', value=1)

    energydf = merged[-1100:-100]
    energyforecast = merged[-100:].drop(columns=['energy_consumption'])

    return energydf, energyforecast
