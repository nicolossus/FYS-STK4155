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


def ridge_shrinkage():
    """
    Ridge
    """
    N = 1000
    sigma2 = 1
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_ridge = Ridge()
    poly_deg = 5
    lamb = np.logspace(1, 4, 15)
    parameters = []

    for i in range(len(lamb)):
        model_ridge.fit(x, z, poly_deg, lamb[i])
        parameters.append(model_ridge.b[1:])
    parameters = np.array(parameters)

    cmap = plt.get_cmap("Greens")
    norm = matplotlib.colors.Normalize(vmin=-10, vmax=model_ridge.params - 1)

    fig = plt.figure(figsize=(8, 6))
    plt.grid()
    for i in range(model_ridge.params - 1):
        plt.plot(np.log(lamb), parameters[:, i], color=cmap(norm(i)))

    plt.plot((np.log(lamb[0]), np.log(lamb[-1])),
             (0, 0), color="black", linewidth=2)
    plt.show()
    fig.savefig(fig_path("ridge_shrinkage.pdf"))


def ridge_model_selection():
    N = 500
    sigma2 = 0.5
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_ridge = Ridge()
    poly_deg = [3, 5, 7, 9]
    k = 5
    lamb = np.logspace(-3, 1, 20)

    mse_test = np.zeros((len(poly_deg), len(lamb)))

    folds = kfold(list(range(N)), k)

    fig = plt.figure()

    for i in range(len(poly_deg)):
        for j in range(len(lamb)):
            for l in range(k):
                train_idx, test_idx = folds(l)
                model_ridge.fit(x[train_idx], z[train_idx],
                                poly_deg[i], lamb[j])
                mse_test[i, j] += model_ridge.mse(x[test_idx], z[test_idx])

            mse_test[i, j] /= k

        plt.plot(np.log10(lamb), mse_test[i])

    plt.grid()
    plt.xlabel("Ridge Penalty ($\\lambda$)")
    plt.ylabel("Test MSE")
    plt.legend([f"p = {poly_deg[i]}" for i in range(len(poly_deg))])
    plt.show()
    fig.savefig(fig_path("ridge_best_model.pdf"))


def ridge_bias_variance():
    N = 1000
    sigma2 = 0.5
    x = np.random.uniform(0, 1, (N, 2))
    z_noiseless = frankeFunction(x[:, 0], x[:, 1])
    z = z_noiseless + np.random.normal(0, sigma2, N)
    poly_deg = 9
    lamb = np.logspace(-3, 1, 15)

    model_ridge = Ridge()
    resamples = 40
    variance = np.zeros(len(lamb))
    bias2 = np.zeros(len(lamb))

    for i in range(len(lamb)):
        predicted = np.zeros((resamples, N))
        for j in range(resamples):
            x_resample = np.random.uniform(0, 1, (N, 2))
            z_resample = frankeFunction(
                x_resample[:, 0], x_resample[:, 1]) + np.random.normal(0, sigma2, N)

            model_ridge.fit(x_resample, z_resample, poly_deg, lamb[i])
            predicted[j] = model_ridge.predict(x)

        variance[i] = np.mean(np.var(predicted, axis=0))
        bias2[i] = np.mean(np.mean((predicted - z_noiseless), axis=0)**2)
    fig = plt.figure()
    plt.grid()
    plt.plot(np.log10(lamb), variance)
    plt.plot(np.log10(lamb), bias2)
    plt.plot(np.log10(lamb), variance + bias2)
    plt.legend(["Variance", "Bias^2", "Variance + Bias^2"])
    plt.show()
    fig.savefig(fig_path("ridge_bias_variance.pdf"))


if __name__ == "__main__":
    ridge_shrinkage()
    ridge_model_selection()
    ridge_bias_variance()
