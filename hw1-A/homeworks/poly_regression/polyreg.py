"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.means: np.ndarray = None
        self.sds: np.ndarray = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        n = len(X)

        P = np.empty((n, degree))
        for j in range(0, degree):
            P[:, j] = X[:, 0] ** (j + 1)

        return P

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # add 1s column and polynomial columns
        X_ = np.c_[np.ones([len(X), 1]), self.polyfeatures(X, self.degree)]

        m = X_.shape[1]
        means = np.zeros(m)
        sds = np.zeros(m)

        # Iterate through columns for standardization
        for i in range(m):
            # Calculate mean and standard deviations
            means[i] = np.mean(X_[:, i])
            sds[i] = np.sqrt(np.mean((X_[:, i] - means[i]) ** 2))

            # Standardize column to z-scores, unless column variance is zero
            X_[:, i] = (X_[:, i] - means[i]) / sds[i] if sds[i] != 0 else X_[:, i]

        # Save column means and standard deviations
        self.means = means
        self.sds = sds

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(m)

        # do not regularize the intercept coefficient
        reg_matrix[0, 0] = 0

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.solve(X_.T @ X_ + reg_matrix, X_.T @ y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # add 1s column and polynomial columns
        X_ = np.c_[np.ones([len(X), 1]), self.polyfeatures(X, self.degree)]

        # standardize data
        for i in range(X_.shape[1]):
            # Standardize column to z-scores, unless column variance is zero
            if self.sds[i] != 0:
                X_[:, i] = (X_[:, i] - self.means[i]) / self.sds[i]

        # calculate predictions for expanded data using the weights
        return X_ @ self.weight



@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    # calculate and return mean squared error
    return (a - b).T @ (a - b) / len(a)


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)

    f = PolynomialRegression(degree, reg_lambda)

    # fill in errorTrain and errorTest arrays
    for i in range(1, n):
        f.fit(Xtrain[0:(i + 1)], Ytrain[0:(i + 1)])
        errorTrain[i] = mean_squared_error(f.predict(Xtrain)[0:(i + 1)], Ytrain[0:(i + 1)])
        errorTest[i] = mean_squared_error(f.predict(Xtest), Ytest)

    return errorTrain, errorTest