import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
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


def cross_validate(model, X, Y, w, K=3, random=42):
    if random:
        cv = KFold(K, shuffle=True, random_state=random)
    else:
        cv = KFold(K, shuffle=True)

    cross_val_mae = []
    for train, val in cv.split(X, Y):
        model.fit(X[train],
                  Y[train])
        y_pred = model.predict(X[val])
        cross_val_mae.append(WMAE(Y[val], y_pred, w[val]))

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


def hold_out_residual(model, x, y, weights,
                      endog_titles=[], test_size=.2):
    x_train, x_test, y_train, y_test, w_train, w_test = \
        train_test_split(x, y, weights, test_size=test_size)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    for col, title in enumerate(endog_titles):
        resid_plot(y_pred[:, col], y_test.iloc[:, col], title)

    return WMAE(y_test.values, y_pred, weights=w_test.values)


def dtwdistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')

    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2 - 1)])


def DTWDistance(s1, s2, w):
    DTW = {}
    w = max(w, abs(len(s1) - len(s2)))
    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w, min(len(s2), i + w))):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum=0
    for ind, i in enumerate(s1):

        lower_bound=min(s2[(ind - r if ind - r>=0 else 0):(ind + r)])
        upper_bound=max(s2[(ind - r if ind - r>=0 else 0):(ind + r)])

        if i>upper_bound:
            LB_sum=LB_sum + (i - upper_bound) ** 2
        elif i<lower_bound:
            LB_sum=LB_sum + (i - lower_bound)**2

    return np.sqrt(LB_sum)
