#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from func import *
import os

# Set seeds
np.random.seed(42)
rd.seed(42)
print(os.getcwd())

if not os.path.exists("./figures"):
    os.makedirs("./figures")

"""
# OLS on varying amounts of data and noise. Calculate training MSE, R2 and CI
# ----------------------------------------------------------------------------
N = [100, 100, 10000, 10000]  # Number of data points
sigma2 = [0.01, 1, 0.01, 1]   # Irreducable error
mse = []
r2 = []
conf_intervals = []
model_ols = OLS()
poly_deg = 5  # complexity
p = 0.9  # 90% confidence interval
for i in range(len(N)):
    x = np.random.uniform(0, 1, (N[i], 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + \
        np.random.normal(0, sigma2[i], N[i])
    model_ols.fit(x, z, poly_deg)
    mse.append(model_ols.mse(x, z))
    r2.append(model_ols.r2(x, z))
    conf_intervals.append(model_ols.confidence_interval(p))


labels = generate_labels(poly_deg)
cmap = plt.get_cmap("Greens")
norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(conf_intervals[0]))

for n in range(len(N)):
    print(
        f"mse={mse[n]:.3f}, r2={r2[n]:.3f} for N={N[n]}, sigma2={sigma2[n]}")
    fig = plt.figure()
    fig.suptitle(f"N = {N[n]}, $\sigma^2$ = {sigma2[n]}")
    plt.yticks(np.arange(model_ols.params), labels)
    plt.grid()

    for i in range(len(conf_intervals[0])):
        plt.plot(conf_intervals[n][i], (i, i), color=cmap(norm(i)))
        plt.plot(conf_intervals[n][i], (i, i), "o", color=cmap(norm(i)))
    fig.savefig(f"./figures/conf_{N[n]}_{sigma2[n]}.pdf")
# ----------------------------------------------------------------------------


# Perform data split and calculate training/testing mse
# ----------------------------------------------------------------------------
N = 300
sigma2 = 1
ratio = 0.25
model_ols = OLS()
poly_deg = 7

x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

train_idx, test_idx = split_data(list(range(N)), ratio=ratio)
model_ols.fit(x[train_idx], z[train_idx], poly_deg)
mse_train = model_ols.mse(x[train_idx], z[train_idx])
mse_test = model_ols.mse(x[test_idx], z[test_idx])
print(
    f"mse_train = {mse_train:.3f} , mse_test = {mse_test:.3f}, for N = {N}" +
    f", sigma2 = {sigma2}, poly_deg = {poly_deg}")
# ----------------------------------------------------------------------------


# Calculate train/test MSE for varying complexity using CV on OLS
# ----------------------------------------------------------------------------
N = [500, 5000]
y_lim = [[0.2, 0.8], [0.26, 0.45]]
repeat = 25
sigma2 = 0.5
model_ols = OLS()
poly_deg_max = 9
mse_train = np.zeros((repeat, poly_deg_max))
mse_test = np.zeros((repeat, poly_deg_max))
k = 5

for n in range(len(N)):  # calculate for small and large dataset
    for r in range(repeat):  # resample to make many models
        x = np.random.uniform(0, 1, (N[n], 2))
        z = frankeFunction(x[:, 0], x[:, 1]) + \
            np.random.normal(0, sigma2, N[n])

        for i in range(poly_deg_max):
            folds = kfold(list(range(N[n])), k=5)

            for j in range(k):
                train_idx, test_idx = folds(j)
                model_ols.fit(x[train_idx], z[train_idx], i)
                mse_train[r, i] += model_ols.mse(x[train_idx], z[train_idx])
                mse_test[r, i] += model_ols.mse(x[test_idx], z[test_idx])

            mse_train[r, i] /= k
            mse_test[r, i] /= k

    fig = plt.figure()
    fig.suptitle(f"Train vs Test MSE, N = {N[n]}, $\sigma^2$ = {sigma2}")
    axes = plt.gca()
    axes.set_ylim(y_lim[n])
    plt.grid()

    plt.plot(np.arange(poly_deg_max), np.mean(
        mse_train, axis=0), color="blue", linewidth=3)
    plt.plot(np.arange(poly_deg_max), np.mean(
        mse_test, axis=0), color="red", linewidth=3)

    for r in range(repeat):
        plt.plot(np.arange(poly_deg_max),
                 mse_train[r], color="blue", alpha=0.1)
        plt.plot(np.arange(poly_deg_max), mse_test[r], color="red", alpha=0.1)

    plt.legend(["train MSE", "test MSE"])
    fig.savefig(f"./figures/train_test_mse_{N[n]}_{sigma2}.pdf")


# ----------------------------------------------------------------------------
"""

# Ridge
# ----------------------------------------------------------------------------
N = 1000
sigma2 = 1
x = np.random.uniform(0, 1, (N, 2))
z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

model_ridge = Ridge()
poly_deg = 2
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

"""
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
"""
