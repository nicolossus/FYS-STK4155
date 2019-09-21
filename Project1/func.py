#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def frankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4


class LinearModel:
    def design_matrix(self, x, poly_deg, intercept=True):
        n = x.shape[0]
        params = int(((poly_deg + 2) * (poly_deg + 1)) / 2) - (not intercept)
        X = np.zeros((n, params))

        idx = 0
        for i in range((not intercept), poly_deg + 1):
            X[:, idx] = x[:, 0]**i
            idx += 1

        for i in range(1, poly_deg + 1):
            for j in range(poly_deg - i + 1):
                X[:, idx] = (x[:, 0]**j) * (x[:, 1]**i)
                idx += 1

        return X, params

    def confidence_interval(p):
        t = stats.t(df=N - self.eff_params).ppf(p)
        self.cinterval = [[self.b[i] - self.b_var[i] * t, b[i] + b_var[i] * t] for
                          i in range(P)]

    def mse(self, x, y):
        n = y.size
        _mse = 1 / n * np.sum((y - self.predict(x))**2)
        return _mse

    def r2(self, x, y):
        n = y.size
        y_ave = np.mean(y)
        _r2 = 1 - np.sum((y - self.predict(x))**2) / np.sum((y - y_ave)**2)
        return _r2


class OLS(LinearModel):
    def fit(self, x, y, poly_deg):
        self.N = x.shape[0]
        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg)

        self.inv_cov_matrix = np.linalg.inv(X.T @ X)
        self.b = self.inv_cov_matrix @ X.T @ y

        self.eff_params = self.params
        self.b_var = np.diag(self.inv_cov_matrix) *\
            self.N / (self.N - self.eff_params) * self.mse(x, y)

    def predict(self, x):
        X, P = self.design_matrix(x, self.poly_deg)
        pred = X @ self.b
        return pred


class Ridge(LinearModel):
    def fit(self, x, y, poly_deg, lamb):
        self.N = x.shape[0]
        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg, intercept=False)
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        X_norm = (X - self.X_mean[np.newaxis, :]) / self.X_std[np.newaxis, :]
        self.inv_cov_matrix = np.linalg.inv(X_norm.T @ X_norm +
                                            lamb * np.identity(self.params))

        self.params += 1
        self.eff_params = np.trace(X_norm @ self.inv_cov_matrix @ X_norm.T) + 1
        self.b = np.zeros(self.params)
        self.b[0] = np.mean(y)
        self.b[1:] = self.inv_cov_matrix @ X_norm.T @ y

        self.b_var = np.zeros(self.params)
        self.b_var[0] = 1 / self.N
        self.b_var[1:] = np.diag(self.inv_cov_matrix @ X.T @ X @ self.inv_cov_matrix)
        self.b_var *= self.N / (self.N - self.eff_params) * self.mse(x, y)

    def predict(self, x):
        X, P = self.design_matrix(x, self.poly_deg, intercept=False)
        X_norm = (X - self.X_mean[np.newaxis, :]) / self.X_std[np.newaxis, :]
        pred = X_norm @ self.b[1:] + self.b[0]
        return pred


def split_data(n, ratio=0.25):
    test_set_size = int(ratio * n)
    indicies = list(range(n))
    rd.shuffle(indicies)
    test_idx = indicies[:test_set_size]
    train_idx = indicies[test_set_size:]
    return train_idx, test_idx


def kfold(n, k=5):
    indicies = list(range(n))
    rd.shuffle(indicies)
    N = ceil(n / k)
    indicies_split = []
    for i in range(k):
        a = i * N
        b = (i + 1) * N
        if b > n:
            b = n
        indicies_split.append(indicies[a:b])

    def folds(i):
        test_idx = indicies_split[i]
        train_idx = indicies_split[:i] + indicies_split[i + 1:]
        train_idx = [item for sublist in train_idx for item in sublist]
        return train_idx, test_idx

    return folds


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    print(x)

    z = frankeFunction(x, y)

    # Plot the surface.

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
