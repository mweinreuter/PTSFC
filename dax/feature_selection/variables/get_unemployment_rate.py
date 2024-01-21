import pandas as pd
import numpy as np
import statsmodels.api as sm


def merge_unemployment_rate(daxdata):

    print('update unemployment rate on: https://statistik.arbeitsagentur.de/DE/Navigation/Statistiken/Interaktive-Statistiken/Zeitreihen/Lange-Zeitreihen-Nav.html')
    unemployment_rate = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/dax/feature_selection/variables/unemployment_rate.csv')
    unemployment_rate['rate'] = unemployment_rate['rate'].str.replace(
        ',', '.').astype(float)

    # predict missing montly rate based on AR(3) model
    unemployment_rate['Lag1'] = unemployment_rate['rate'].shift(1)
    unemployment_rate['Lag2'] = unemployment_rate['rate'].shift(2)
    unemployment_rate['Lag3'] = unemployment_rate['rate'].shift(3)

    # train model with all values not containing nans
    bool_series = unemployment_rate['rate'].isna()
    first_index = bool_series.idxmax()
    unemployment_rate_model = unemployment_rate[3:first_index]

    # get data
    X_lags_ext = sm.add_constant(
        unemployment_rate_model.loc[:, 'Lag1':'Lag3'])
    y_index_ext = np.array(unemployment_rate_model.loc[:, 'rate'])

    # AR model (using OLS)
    model = sm.OLS(y_index_ext, X_lags_ext).fit()

    # Get the beta coefficients
    betas = np.array(model.params)

    # predict future prod indexes
    X_array = np.array([1, unemployment_rate.at[first_index, 'Lag1'], unemployment_rate.at[first_index, 'Lag2'],
                        unemployment_rate.at[first_index, 'Lag3']])
    lag1, lag2 = np.nan, np.nan

    for idx, row in unemployment_rate[first_index:].iterrows():
        # predict and safe
        index_pred = betas.dot(X_array)
        unemployment_rate.at[idx, 'rate'] = index_pred

        # update data for next iteration
        lag1, lag2 = X_array[1], X_array[2]
        X_array[2], X_array[3] = lag1, lag2  # shift backwards
        X_array[1] = index_pred

    daxdata['year'] = daxdata.index.year
    daxdata['month'] = daxdata.index.month
    daxdata = daxdata.reset_index()

    merged = pd.merge(daxdata, unemployment_rate, how='left', left_on=['year', 'month'], right_on=[
                      'year', 'month']).set_index('Date').drop(columns={'year', 'month', 'Lag1', 'Lag2',
                                                                        'Lag3'})

    return merged, model.rsquared_adj
