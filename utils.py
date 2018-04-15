import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from statsmodels.tsa.stattools import adfuller


# def dickey_fuller_test(timeseries, window=12):
#     # code snagged from: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#     # Determing rolling statistics
#     rolmean = timeseries.rolling(window=window, center=False).mean()
#     rolstd = timeseries.rolling(window=window, center=False).std()

#     # Plot rolling statistics:
#     plt.plot(timeseries, color='blue', label='Original')
#     plt.plot(rolmean, color='red', label='Rolling Mean')
#     plt.plot(rolstd, color='black', label='Rolling Std')
#     plt.legend(loc='best')
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show(block=False)
#     # Perform Dickey-Fuller test:
#     print('Results of Dickey-Fuller Test:')
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s)' % key] = value
#     print(dfoutput)


def WMAE(y_pred, y_true, weights=None):
    if weights:
        return np.mean(np.abs(y_pred - y_true) * weights)
    return np.mean(np.abs(y_pred - y_true))
