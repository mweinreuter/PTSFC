import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm

from dax.help_functions.get_dax_data import get_prepared_data
from evaluation.help_functions.prepare_data import next_working_days
from dax.models.ARMA_NN.distance import calculate_distance

# Suppress warnings
# Note: I always checked that results fit (iterative testing, thereby, warnings can be ignored for now)
warnings.filterwarnings(
    "ignore", message="An unsupported index was provided and will be ignored", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="A date index has been provided, but it has no associated frequency information", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="No supported index is available. Prediction results will be given with an integer index beginning at `start`.", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.", category=FutureWarning)
warnings.filterwarnings(
    "ignore", message="Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.", category=UserWarning)
warnings.filterwarnings(
    "ignore", message="Non-invertible starting MA parameters found. Using zeros as starting parameters.", category=UserWarning)

# optimal ARIMA order based on bic
opt_orders = {
    1: (0, 0, 0),
    2: (5, 0, 0),
    3: (5, 0, 1),
    4: (1, 0, 0),
    5: (5, 0, 2)
}

# adjust parameters if needed
k = 50
m = 9


def get_arma_nn_forecasts15(daxdata=pd.DataFrame(), quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], submission=True):

    if daxdata.empty:
        daxdata = get_prepared_data()
    daxdata = daxdata.reset_index()

    horizons = [1, 2, 3, 4, 5]

    # store arima means in dict
    arima_means = {}

    # store forecast corrections in dict
    forecast_correction = {}

    # store corrected means and residual variance in list
    corrected_means = []
    variances = []

    for h in horizons:

        # estimate mean with ARIMA
        model = ARIMA(daxdata[[f'LogRetLag{h}']], order=opt_orders[h])
        model_fit = model.fit()
        model_residuals = model_fit.resid
        variances.append(np.var(model_residuals))

        # get col index of  observed Log Return
        ci_logret = daxdata.columns.get_loc(f'LogRetLag{h}')

        # forecast desired horizon
        mi = daxdata.index.max()
        arima_mean = model_fit.predict(start=mi+1, end=mi+6).iloc[(h-1)]
        arima_means.update({h: arima_mean})

        # calculate distances for return
        daxdata[f'distance_{h}'] = np.nan
        starting_index = daxdata.index.min() + k

        # get col index
        ci = daxdata.columns.get_loc(f'distance_{h}')

        for j in range(k, len(daxdata)):
            daxdata.iat[j, ci] = calculate_distance(daxdata, h, j, k)
        daxdata = daxdata.dropna(subset=[f'distance_{h}'])

        # set last value (zero distance) high enough so that index does not get chosen
        daxdata.iat[len(daxdata)-1, ci] = 10

        # top m matching parts
        top_indices = daxdata[f'distance_{h}'].nsmallest(m).index

        # iterate over top m matching parts, estimate prediction and store residual
        residuals = []
        for ti in top_indices:

            if (ti < len(daxdata)-h):

                # train model with all data up to ti
                model = ARIMA(
                    daxdata[[f'LogRetLag{h}']][:ti+1], order=opt_orders[h])
                model_fit = model.fit()

                # get col index of  observed Log Return
                ci_logret = daxdata.columns.get_loc(f'LogRetLag{h}')

                # calculate residual for desired horizon
                yhat = model_fit.predict(start=ti+1, end=ti+6).iloc[(h-1)]
                yobs = daxdata.iat[ti+h-1, ci_logret]
                residual = yobs - yhat

                # append to list of residuals
                residuals.append(residual)

        forecast_correction.update({h: np.mean(np.array(residuals))})

        # calculate corrected mean
        corrected_means.append(arima_means[h]-forecast_correction[h])

    # get quantiles
    column_names = [f'q{q}' for q in quantiles]
    daxdata = daxdata.set_index("Date")
    start_date_dates = max(daxdata.index).strftime('%Y-%m-%d')
    dates = next_working_days(start_date_dates, 5)
    quantile_df = pd.DataFrame(index=dates, columns=column_names)

    # calculate prediction quantiles
    for h in range(0, 5):

        mean = corrected_means[h]
        variance = variances[h]

        for q in quantiles:
            quantile_q = mean + variance*norm.ppf(q)
            quantile_df.loc[dates[h]][f'q{q}'] = quantile_q

    if submission == True:
        date_st = daxdata.index[-1].strftime('%Y-%m-%d')
        quantile_df.insert(0, 'forecast_date', date_st)
        quantile_df.insert(1, 'target', 'DAX')
        quantile_df.insert(
            2, "horizon", [str(i) + " day" for i in (1, 2, 5, 6, 7)])
        quantile_df.index.name = "date_time"

    return quantile_df


warnings.resetwarnings()
