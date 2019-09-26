#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso


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
            for j in range(i + 1):
                X[:, idx] = x[:, 0]**(i - j) * x[:, 1]**j
                idx += 1

        return X, params

    def normalize_design_matrix(self, X):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

        X_norm = (X - self.X_mean[np.newaxis, :]) / self.X_std[np.newaxis, :]
        return X_norm

    def confidence_interval(self, p):
        t = stats.t(df=self.N - self.eff_params).ppf(2 * p - 1)
        conf_intervals = [[self.b[i] - self.b_var[i] * t, self.b[i] + self.b_var[i] * t] for
                          i in range(self.params)]
        return conf_intervals

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
        X, self.params = self.design_matrix(x, poly_deg, intercept=False)
        X = self.normalize_design_matrix(X)

        self.inv_cov_matrix = np.linalg.pinv(X.T @ X)

        self.params += 1
        self.eff_params = np.trace(X @ self.inv_cov_matrix @ X.T) + 1
        self.b = np.zeros(self.params)
        self.b[0] = np.mean(y)
        self.b[1:] = self.inv_cov_matrix @ X.T @ y

        self.eff_params = self.params
        self.b_var = np.zeros(self.params)
        self.b_var[0] = 1 / self.N
        self.b_var[1:] = np.diag(self.inv_cov_matrix)
        self.b_var *= self.N / (self.N - self.eff_params) * self.mse(x, y)

    def predict(self, x):
        X, P = self.design_matrix(x, self.poly_deg, intercept=False)
        X = self.normalize_design_matrix(X)
        pred = X @ self.b[1:] + self.b[0]
        return pred


class Ridge(LinearModel):
    def fit(self, x, y, poly_deg, lamb):
        self.N = x.shape[0]
        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg, intercept=False)
        X = self.normalize_design_matrix(X)

        self.inv_cov_matrix = np.linalg.pinv(
            X.T @ X + lamb * np.identity(self.params))

        self.params += 1
        self.eff_params = np.trace(X @ self.inv_cov_matrix @ X.T) + 1
        self.b = np.zeros(self.params)
        self.b[0] = np.mean(y)
        self.b[1:] = self.inv_cov_matrix @ X.T @ y

        self.b_var = np.zeros(self.params)
        self.b_var[0] = 1 / self.N
        self.b_var[1:] = np.diag(self.inv_cov_matrix @
                                 X.T @ X @ self.inv_cov_matrix)
        self.b_var *= self.N / (self.N - self.eff_params) * self.mse(x, y)

    def predict(self, x):
        X, P = self.design_matrix(x, self.poly_deg, intercept=False)
        X_norm = (X - self.X_mean[np.newaxis, :]) / self.X_std[np.newaxis, :]
        pred = X_norm @ self.b[1:] + self.b[0]
        return pred


class MyLasso(LinearModel):
    def fit(self, x, y, poly_deg, lamb):
        self.N = x.shape[0]
        self.poly_deg = poly_deg
        X, self.params = self.design_matrix(x, poly_deg, intercept=False)

        self.params += 1
        self.lasso = Lasso(alpha=lamb, fit_intercept=True, max_iter=1000000)
        self.lasso.fit(X, y)
        self.b = np.zeros(self.params)
        self.b[0] = self.lasso.intercept_
        self.b[1:] = self.lasso.coef_

    def predict(self, x):
        X, P = self.design_matrix(x, self.poly_deg, intercept=False)
        pred = self.lasso.predict(X)
        return pred


def split_data(indicies, ratio=0.25):
    n = len(indicies)
    test_set_size = int(ratio * n)
    rd.shuffle(indicies)
    test_idx = indicies[:test_set_size]
    train_idx = indicies[test_set_size:]
    return train_idx, test_idx


def generate_labels(N):
    labels = []

    for i in range(N + 1):
        for j in range(i + 1):
            label = f"x^{i-j} \\cdot y^{j}"
            label = label.replace("x^0 \\cdot y^0", "1")
            label = label.replace("x^0 \\cdot", "")
            label = label.replace("\\cdot y^0", "")
            label = label.replace("x^1", "x")
            label = label.replace("y^1", "y")
            label = "$" + label + "$"
            labels.append(label)
    return labels


def kfold(indicies, k=5):
    n = len(indicies)
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


def down_sample(terrain, N):
    m, n = terrain.shape
    m_new, n_new = int(m / N), int(n / N)
    terrain_new = np.zeros((m_new, n_new))
    for i in range(m_new):
        for j in range(n_new):
            slice = terrain[N * i:N * (i + 1), N * j:N * (j + 1)]
            terrain_new[i, j] = 1 / (slice.size)**2 * np.sum(slice)

    return terrain_new


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
