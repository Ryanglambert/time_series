import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
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

def WMAE(y_true, y_pred, weights=None):
    if weights is None:
        # don't apply weights
        return np.mean(np.abs(y_true - y_pred))
    # do apply weights
    return np.mean(np.abs(y_true - y_pred) * weights)


def cross_validate(model, X, Y, w, K=3):
    cv = KFold(K, shuffle=True, random_state=42)

    cross_val_mae = []
    for train, val in cv.split(X, Y):
        model = Pipeline(steps=[('minmax_scaler', MinMaxScaler()),
                                ('RANSAC', RANSACRegressor())])
        model.fit(X.iloc[train],
                  Y.iloc[train])
        y_pred = model.predict(X.iloc[val])
        cross_val_mae.append(WMAE(Y.iloc[val].values, y_pred, w.iloc[val].values))

    print('Cross Val Score: {}\n Cross Val Std: {}\n K: {}'.format(np.mean(cross_val_mae), np.std(cross_val_mae), K))


def resid_plot(y_pred, y_actual, title):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(y_pred, y_pred - y_actual)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Predicted - Actual')
    plt.title(title + " Residuals", fontsize=10)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(y_pred - y_actual, orientation='horizontal', bins=15)
    plt.title(title + " Hist of Residuals", fontsize=10)
#     plt.savefig('/Users/ryanlambert/Desktop/' + title + "_Hist_of_Residuals.png")
    plt.show()
    plt.hexbin(y_pred, y_pred - y_actual, gridsize=20)
    plt.title(title + " Hexbin of Residuals", fontsize=10)
