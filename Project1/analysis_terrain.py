# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random as rd
import sys

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

from func import *
from setup import *

# Set seed
np.random.seed(42)
rd.seed(42)

terrain_downsample = down_sample(terrain1, 10)

m, n = terrain_downsample.shape
z = np.ravel(terrain_downsample)
x = np.array([j for i in range(m) for j in n * [i]]) / m
y = np.array(m * list(range(n))) / n

x = np.array([[i, j] for i, j in zip(x, y)])

model_ridge = Ols()
model_ridge.fit(x, z, 80, lamb=0)

terrain_fitted = model_ridge.predict(x)
terrain_fitted = np.reshape(terrain_fitted, (m, n))

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_fitted, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
