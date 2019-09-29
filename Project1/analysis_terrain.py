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

terrain_downsample = down_sample(terrain1, 20)

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_downsample, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

m, n = terrain_downsample.shape
z = np.ravel(terrain_downsample)
print(z[50])
x = np.array([j for i in range(m) for j in n * [i]]) / m
y = np.array(m * list(range(n))) / n

x = np.array([[i, j] for i, j in zip(x, y)])
"""
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

plt.plot(poly_deg, mse_ols)
plt.show()
# ----------------------------------------------------------------------------
"""
# Ridge
# ----------------------------------------------------------------------------
k = 5
N = len(y)
folds = kfold(list(range(N)), k)

lamb = np.logspace(-2, 2, 4)
poly_deg = [3, 5, 7]

mse_ridge = np.zeros((len(lamb), len(poly_deg)))

for i in range(len(poly_deg)):
    for l in range(len(lamb)):
        for j in range(k):
            train_idx, test_idx = folds(j)
            model_ridge = Ridge()
            model_ridge.fit(x[train_idx], z[train_idx], poly_deg[i], lamb[l])
            mse_ridge[l, i] += model_ridge.mse(x[test_idx], z[test_idx])

            mse_ridge[l, i] /= k

plt.plot(np.log10(lamb), mse_ridge)
plt.show()
# ----------------------------------------------------------------------------
