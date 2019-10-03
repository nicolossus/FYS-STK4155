# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random as rd
import sys

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from func import *
from setup import *

# Set seed
np.random.seed(42)
rd.seed(42)

terrain1 = imread("SRTM_data_Norway_1.tif")


plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

terrain_downsample = down_sample(terrain1, 80)

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_downsample, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

m, n = terrain_downsample.shape
z = np.ravel(terrain_downsample)
x = np.array([j for i in range(m) for j in n * [i]]) / m
y = np.array(m * list(range(n))) / n

x = np.array([[i, j] for i, j in zip(x, y)])

# OLS
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)
poly_deg_max = 15

poly_deg = np.arange(1, poly_deg_max)

mse_ols = np.zeros(len(poly_deg))

for i in range(len(poly_deg)):
    for j in range(k):
        train_idx, test_idx = folds(j)
        model_ols = OLS()
        model_ols.fit(x[train_idx], z[train_idx], poly_deg[i])
        mse_ols[i] += model_ols.mse(x[test_idx], z[test_idx])

    mse_ols[i] /= k
fig = plt.figure()
plt.plot(poly_deg, mse_ols)
fig.savefig(fig_path("terrain_ols_best_model.pdf"))
plt.show()
# ----------------------------------------------------------------------------


# Ridge
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)

lamb = np.logspace(-15, 0, 10)
poly_deg = [3, 5, 7, 10, 15, 20, 30]

mse_ridge = np.zeros((len(lamb), len(poly_deg)))

for i in range(len(poly_deg)):
    for l in range(len(lamb)):
        for j in range(k):
            train_idx, test_idx = folds(j)
            model_ridge = Ridge()
            model_ridge.fit(x[train_idx], z[train_idx], poly_deg[i], lamb[l])
            mse_ridge[l, i] += model_ridge.mse(x[test_idx], z[test_idx])

            mse_ridge[l, i] /= k
fig = plt.figure()
plt.plot(poly_deg, mse_ridge)
fig.savefig(fig_path("terrain_ridge_best_model.pdf"))
plt.show()
plt.show()
# ----------------------------------------------------------------------------


# Lasso
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)

#lamb = np.logspace(-15, 0, 10)
#poly_deg = [3, 5, 7, 10, 15, 20, 30, 40]

lamb = np.logspace(-2.5, 0, 6)
poly_deg = [3, 5, 7, 10, 15, 20, 30]


mse_lasso = np.zeros((len(lamb), len(poly_deg)))

for i in range(len(poly_deg)):
    for l in range(len(lamb)):
        model_lasso = MyLasso()
        for j in range(k):
            train_idx, test_idx = folds(j)
            model_lasso.fit(x[train_idx], z[train_idx], poly_deg[i], lamb[l])
            mse_lasso[l, i] += model_lasso.mse(x[test_idx], z[test_idx])

            mse_lasso[l, i] /= k
fig = plt.figure()
plt.plot(poly_deg, mse_lasso)
fig.savefig(fig_path("terrain_lasso_best_model.pdf"))
plt.show()

plt.plot(np.log10(lamb), mse_lasso)
plt.show()
# ----------------------------------------------------------------------------
