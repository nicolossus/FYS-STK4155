#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from func import *

# Set seed for debugging purposes
np.random.seed(42)
rd.seed(42)


# OLS on varying amounts of data and noise. Calculate training MSE, R2 and CI
# ----------------------------------------------------------------------------
N = [100, 100, 10000, 10000]  # Number of data points
sigma2 = [0.01, 1, 0.01, 1]   # Irreducable error
mse = []
r2 = []
conf_intervals = []
model_ols = OLS()
poly_deg = 5
p = 0.9
for i in range(len(N)):
    x = np.random.uniform(0, 1, (N[i], 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + \
        np.random.normal(0, sigma2[i], N[i])
    model_ols.fit(x, z, poly_deg)
    mse.append(model_ols.mse(x, z))
    r2.append(model_ols.r2(x, z))
    conf_intervals.append(model_ols.confidence_interval(p))
# ----------------------------------------------------------------------------


# Perform data split and calculate training/testing mse
# ----------------------------------------------------------------------------
N = 1000
sigma2 = 1
ratio = 0.25
model_ols = OLS()
poly_deg = 6

x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

train_idx, test_idx = split_data(list(range(N)), ratio=ratio)
model_ols.fit(x[train_idx], z[train_idx], poly_deg)
mse_train = model_ols.mse(x[train_idx], z[train_idx])
mse_test = model_ols.mse(x[test_idx], z[test_idx])
print(mse_train, mse_test)
# ----------------------------------------------------------------------------


# Calculate train/test MSE for varying complexity using CV on OLS
# ----------------------------------------------------------------------------
N = 1000
sigma2 = 1
model_ols = OLS()
poly_deg_max = 9
mse_train = np.zeros(poly_deg_max,)
mse_test = np.zeros(poly_deg_max)
k = 5

x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

for i in range(poly_deg_max):
    folds = kfold(list(range(N)), k=5)

    for j in range(k):
        train_idx, test_idx = folds(j)
        model_ols.fit(x[train_idx], z[train_idx], i)
        mse_train[i] += model_ols.mse(x[train_idx], z[train_idx])
        mse_test[i] += model_ols.mse(x[test_idx], z[test_idx])

    mse_train[i] /= k
    mse_test[i] /= k

plt.plot(np.arange(poly_deg_max), mse_train)
plt.plot(np.arange(poly_deg_max), mse_test)
plt.legend(["Training MSE", "Test MSE"])
plt.show()

N = 10000
x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

for i in range(poly_deg_max):
    folds = kfold(list(range(N)), k=5)

    for j in range(k):
        train_idx, test_idx = folds(j)
        model_ols.fit(x[train_idx], z[train_idx], i)
        mse_train[i] += model_ols.mse(x[train_idx], z[train_idx])
        mse_test[i] += model_ols.mse(x[test_idx], z[test_idx])

    mse_train[i] /= k
    mse_test[i] /= k

plt.plot(np.arange(poly_deg_max), mse_train)
plt.plot(np.arange(poly_deg_max), mse_test)
plt.legend(["Training MSE", "Test MSE"])
plt.show()
# ----------------------------------------------------------------------------


# Ridge
# ----------------------------------------------------------------------------
N = 1000
sigma2 = 1
x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

model_ridge = Ridge()
poly_deg_max = 9
poly_deg = np.arange(1, poly_deg_max + 1)
lamb = [1e-2, 1e-1, 1e0, 1e1]
mse_train = np.zeros((len(lamb), len(poly_deg)))
mse_test = np.zeros((len(lamb), len(poly_deg)))
k = 5

for l in range(len(lamb)):
    for i in range(len(poly_deg)):
        folds = kfold(list(range(N)), k=5)

        for j in range(k):
            train_idx, test_idx = folds(j)
            model_ridge.fit(x[train_idx], z[train_idx],
                            poly_deg[i], lamb[l])
            mse_train[l, i] += model_ridge.mse(x[train_idx], z[train_idx])
            mse_test[l, i] += model_ridge.mse(x[test_idx], z[test_idx])

        mse_train[l, i] /= k
        mse_test[l, i] /= k

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[0])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[0])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[1])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[1])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[2])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[2])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[3])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[3])
plt.legend(["Training MSE", "Test MSE"])
plt.show()
# ----------------------------------------------------------------------------


# Lasso
# ----------------------------------------------------------------------------
N = 1000
sigma2 = 1
x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

model_lasso = MyLasso()
poly_deg_max = 9
poly_deg = np.arange(1, poly_deg_max + 1)
lamb = [1e-2, 1e-1, 1e0, 1e1]
mse_train = np.zeros((len(lamb), len(poly_deg)))
mse_test = np.zeros((len(lamb), len(poly_deg)))
k = 5

for l in range(len(lamb)):
    for i in range(len(poly_deg)):
        folds = kfold(list(range(N)), k=5)

        for j in range(k):
            train_idx, test_idx = folds(j)
            model_lasso.fit(x[train_idx], z[train_idx], poly_deg[i], lamb[l])
            mse_train[l, i] += model_lasso.mse(x[train_idx], z[train_idx])
            mse_test[l, i] += model_lasso.mse(x[test_idx], z[test_idx])

        mse_train[l, i] /= k
        mse_test[l, i] /= k

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[0])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[0])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[1])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[1])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[2])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[2])
plt.legend(["Training MSE", "Test MSE"])
plt.show()

plt.plot(np.arange(1, poly_deg_max + 1), mse_train[3])
plt.plot(np.arange(1, poly_deg_max + 1), mse_test[3])
plt.legend(["Training MSE", "Test MSE"])
plt.show()
# ----------------------------------------------------------------------------
