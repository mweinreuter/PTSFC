import pandas as pd
import numpy as np


def drop_years(X, X_pred):

    reference_year_lower = X.index.year[0]
    year_upper = X_pred.index.year[-1]

    years_to_drop = []
    years_dict = {2016: 'year_2016',
                  2017: 'year_2017',
                  2018: 'year_2018',
                  2019: 'year_2019',
                  2020: 'year_2020',
                  2021: 'year_2021',
                  2022: 'year_2022',
                  2023: 'year_2023',
                  2024: 'year_2024'
                  }

    for key in years_dict:
        if key <= reference_year_lower:
            years_to_drop.append(years_dict[key])
        elif key > year_upper:
            years_to_drop.append(years_dict[key])

    X = X.drop(columns=years_to_drop)
    X_pred = X_pred.drop(columns=years_to_drop)

    return X, X_pred
