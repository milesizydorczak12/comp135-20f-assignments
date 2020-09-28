# performance_metrics.py
# By: Miles Izydorczak
# Date: 22 September 2020
# Purpose: Exports calc_mean_squared_error and calc_mean_absolute_error
#          helper functions.
#

import numpy as np


def calc_mean_squared_error(y_N, yhat_N):
    ''' Compute the mean squared error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mse : scalar float
        Mean squared error performance metric
        .. math:
            mse(y, \hat{y}) = \frac{1}{N} \sum_{n=1}^N (y_n - \hat{y}_n)^2

    Examples
    --------
    >>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
    >>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
    >>> calc_mean_squared_error(y_N, yhat_N)
    1.3333333333333333
    '''
    mse = 0.0
    N = y_N.shape[0]
    for y in range(N):
        difference = y_N[y] - yhat_N[y]
        difference_squared = difference ** 2
        mse += difference_squared
    if N > 0:
        mse = mse / float(N)
    return mse 


def calc_mean_absolute_error(y_N, yhat_N):
    ''' Compute the mean absolute error given true and predicted values

    Args
    ----
    y_N : 1D array, shape (N,)
        Each entry represents 'ground truth' numeric response for an example
    yhat_N : 1D array, shape (N,)
        Each entry representes predicted numeric response for an example

    Returns
    -------
    mae : scalar float
        Mean absolute error performance metric
        .. math:
            mae(y, \hat{y}) = \frac{1}{N} \sum_{n=1}^N | y_n - \hat{y}_n |

    Examples
    --------
    >>> y_N = np.asarray([-2, 0, 2], dtype=np.float64)
    >>> yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
    >>> calc_mean_absolute_error(y_N, yhat_N)
    0.6666666666666666
    '''

    mae = 0.0
    N = y_N.shape[0]
    for y in range(N):
        difference = abs(y_N[y] - yhat_N[y])
        mae += difference
    if (N > 0):
        mae = mae / float(N)

    return mae 

#### UNIT TESTING ####

# y_N = np.asarray([-2, 0, 2], dtype=np.float64)
# yhat_N = np.asarray([-4, 0, 2], dtype=np.float64)
# mae = calc_mean_absolute_error(y_N, yhat_N)
# print('MAE 1: ', mae)
# mse = calc_mean_squared_error(y_N, yhat_N)
# print('MSE 1: ', mse)

# y_N = np.asarray([1, 3, 10], dtype=np.float64)
# yhat_N = np.asarray([1, 0, 10], dtype=np.float64)
# mae = calc_mean_absolute_error(y_N, yhat_N)
# print('MAE 2: ', mae)
# mse = calc_mean_squared_error(y_N, yhat_N)
# print('MSE 2: ', mse)

# y_N = np.asarray([5, 3, 10], dtype=np.float64)
# yhat_N = np.asarray([1, 0, 10], dtype=np.float64)
# mae = calc_mean_absolute_error(y_N, yhat_N)
# print('MAE 3: ', mae)
# mse = calc_mean_squared_error(y_N, yhat_N)
# print('MSE 3: ', mse)

# y_N = np.asarray([], dtype=np.float64)
# yhat_N = np.asarray([], dtype=np.float64)
# mae = calc_mean_absolute_error(y_N, yhat_N)
# print('MAE 3: ', mae)
# mse = calc_mean_squared_error(y_N, yhat_N)
# print('MSE 3: ', mse)
