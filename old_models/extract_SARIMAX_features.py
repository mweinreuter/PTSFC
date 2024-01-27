import pandas as pd

from energy_consumption.feature_selection.extract.extract_energy_data import get_data
from energy_consumption.feature_selection.collect import dummy_mapping, weather_sunhours, production_index


def get_energy_and_SARIMAX_features(energydata=pd.DataFrame):

    # SARIMAX trains model based on data one month ago
    month_year = 1/12

    if energydata.empty:
        energydata = get_data(num_years=month_year)

    else:
        energydata = energydata[-564:]

    energydata = (energydata
                  .pipe(dummy_mapping.get_day_mapping)
                  .pipe(dummy_mapping.get_holiday_mapping)
                  .pipe(weather_sunhours.ec_sun_hours_merge)
                  .pipe(production_index.merge_production_indexes)
                  )

    return energydata
