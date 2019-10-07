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


fig = plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
fig.savefig(fig_path("original_terrain.pdf"))
terrain_downsample = down_sample(terrain1, 80)

fig = plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_downsample, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
fig.savefig(fig_path("downsample_terrain.pdf"))

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
poly_deg_max = 10

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
plt.xlabel("Model Complexity")
plt.ylabel("MSE")
fig.savefig(fig_path("terrain_ols_best_model.pdf"))

# ----------------------------------------------------------------------------


# Ridge
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)

lamb = np.logspace(-6.5, 2, 15)
poly_deg = [4, 5, 6, 7, 9, 11]

mse_ridge = np.zeros((len(lamb), len(poly_deg)))

for i in range(len(poly_deg)):
    for l in range(len(lamb)):
        model_ridge = Ridge()
        for j in range(k):
            train_idx, test_idx = folds(j)

            model_ridge.fit(x[train_idx], z[train_idx], poly_deg[i], lamb[l])
            mse_ridge[l, i] += model_ridge.mse(x[test_idx], z[test_idx])

        mse_ridge[l, i] /= k
fig = plt.figure()
plt.grid()
plt.plot(np.log10(lamb), mse_ridge)
plt.xlabel("$\\log10(\\lambda)$")
plt.ylabel("MSE")
plt.legend(poly_deg)
fig.savefig(fig_path("terrain_ridge_best_model.pdf"))

# ----------------------------------------------------------------------------


# Lasso
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)

# lamb = np.logspace(-15, 0, 10)
# poly_deg = [3, 5, 7, 10, 15, 20, 30, 40]

lamb = np.logspace(-1.5, 0.5, 10)
poly_deg = [4, 5, 6, 7, 9, 11]


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
plt.plot(np.log10(lamb), mse_lasso)
plt.grid()
plt.xlabel("$\\log10(\\lambda)$")
plt.ylabel("MSE")
plt.legend(poly_deg)
fig.savefig(fig_path("terrain_lasso_best_model.pdf"))

# ----------------------------------------------------------------------------
