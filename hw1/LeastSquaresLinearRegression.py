# LeastSquaresLinearRegression.py
# By: Miles Izydorczak
# Date: 22 September 2020
# Purpose:
#

import numpy as np
# No other imports allowed!

class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
        
        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        '''
        N, F = x_NF.shape
        # print("N:", N, "F:", F)
        xtilde_NF = np.hstack([x_NF, np.ones((N, 1))])
        # print("xtilde_NF:", xtilde_NF)

        xTx_FF = np.dot(xtilde_NF.T, xtilde_NF)
        # print("xTx_FF:", xTx_FF)

        inv_xTx_FF = np.linalg.inv(xTx_FF)

        self.w_F = np.dot(inv_xTx_FF, np.dot(xtilde_NF.T, y_N[:,np.newaxis]))
        # print("self.w_F with bias", self.w_F)
        self.b = self.w_F[-1]
        self.w_F = self.w_F[:-1, 0]

        # print("self.w_F without bias", self.w_F)
        # print("bias", self.b)

    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_MF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_M : 1D array, size M
            Each value is the predicted scalar for one example
        '''

        M, F = x_MF.shape
        yhat_M = np.zeros(M)

        yhat_M += self.b
        # print(yhat_M)
        # print(self.w_F.shape)
        # self.w_F = self.w_F[np.newaxis, :]
        # print(self.w_F.shape)
        # print(yhat_M.shape)
        # print(x_MF.shape)

        temp = x_MF * self.w_F[np.newaxis, :]

        temp = np.sum(temp, axis=1)

        # print(temp)
        # print(temp.shape)

        yhat_M += temp

        # print(yhat_M)

        return yhat_M


if __name__ == '__main__':
    # Simple example use case
    # With toy dataset with N=100 examples
    # created via a known linear regression model plus small noise

    prng = np.random.RandomState(0)
    N = 100

    true_w_F = np.asarray([1.1, -2.2, 3.3])
    true_b = 0.0
    x_NF = prng.randn(N, 3)
    y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

    linear_regr = LeastSquaresLinearRegressor()
    linear_regr.fit(x_NF, y_N)
    print("Intercept: ",linear_regr.b)
    print("Weights: ", linear_regr.w_F)

    yhat_N = linear_regr.predict(x_NF)
    print(yhat_N)
