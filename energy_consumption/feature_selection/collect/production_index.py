import pandas as pd
import numpy as np
import statsmodels.api as sm


def merge_production_indexes(energydata):

    productionindexes = pd.read_csv(
        'C:/Users/Maria/Documents/Studium/Pyhton Projekte/PTSFC/energy_consumption/feature_selection/data/prod_index.csv')

    # predict missing montly production indices based on AR(3) model with seasonal dummies (12)
    productionindexes['Lag1'] = productionindexes['index'].shift(1)
    productionindexes['Lag2'] = productionindexes['index'].shift(2)
    productionindexes['Lag3'] = productionindexes['index'].shift(3)
    productionindexes['Lag12'] = productionindexes['index'].shift(12)

    # train model with all values not containing nans
    bool_series = productionindexes['index'].isna()
    first_index = bool_series.idxmax()
    productionindexes_model = productionindexes[12:first_index]

    # get data
    X_lags_ext = sm.add_constant(
        productionindexes_model.loc[:, 'Lag1':'Lag12'])
    y_index_ext = productionindexes_model.loc[:, 'index']

    # AR model (using OLS)
    model = sm.OLS(y_index_ext, X_lags_ext).fit()

    # Get the beta coefficients
    betas = np.array(model.params)

    # predict future prod indexes
    X_array = np.array([1, productionindexes.at[first_index, 'Lag1'], productionindexes.at[first_index, 'Lag2'],
                        productionindexes.at[first_index, 'Lag3'], productionindexes.at[first_index, 'Lag12']])
    lag1, lag2 = np.nan, np.nan

    for idx, row in productionindexes[first_index:].iterrows():
        # predict and safe
        index_pred = betas.dot(X_array)
        productionindexes.at[idx, 'index'] = index_pred

        # update data for next iteration
        lag1, lag2 = X_array[1], X_array[2]
        X_array[2], X_array[3] = lag1, lag2  # shift backwards
        X_array[1] = index_pred

        if idx+2 < len(productionindexes):
            X_array[4] = productionindexes.at[idx+1, 'Lag12']

    energydata['year'] = energydata.index.year
    energydata['month'] = energydata.index.month
    energydata = energydata.reset_index()

    merged = pd.merge(energydata, productionindexes, how='left', left_on=['year', 'month'], right_on=[
                      'year', 'month']).set_index('date_time').drop(columns={'year', 'month', 'Lag1', 'Lag2',
                                                                             'Lag3', 'Lag12', 'index_cleaned'})

    return merged, model.rsquared_adj
