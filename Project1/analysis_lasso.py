# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from func import *
from setup import *

# Set seed
np.random.seed(42)
rd.seed(42)


def lasso_shrinkage():
    """
    Calculate and plot the parameters for Lasso for various penalties
    """
    N = 1000  # Number of data pints
    sigma2 = 0.01  # Irreducable error
    # Generating data
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_lasso = MyLasso()
    poly_deg = 5
    lamb = np.logspace(-4, -1, 15)  # penalty
    parameters = []

    for i in range(len(lamb)):
        model_lasso.fit(x, z, poly_deg, lamb[i])
        parameters.append(model_lasso.b)
    parameters = np.array(parameters)

    cmap = plt.get_cmap("nipy_spectral_r")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=model_lasso.params - 1)

    fig = plt.figure(figsize=(8, 6))
    plt.grid()
    for i in range(model_lasso.params - 1):
        plt.plot(np.log10(lamb), parameters[:, i], color=cmap(norm(i)))

    plt.plot((np.log10(lamb[0]), np.log10(lamb[-1])),
             (0, 0), color="black", ls='--', lw=2)

    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.gca().set_ylabel("Coefficients $\\beta_j$ ")
    plt.gca().set_title("Method: Lasso w/o Resampling")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=0, vmax=model_lasso.params - 1))
    plt.colorbar(sm)
    fig.savefig(fig_path("lasso_shrinkage.pdf"))


def lasso_model_selection():
    """
    Calculate the test MSE of Lasso for various complexitis and penalties
    """
    N = 500  # Number of data points
    sigma2 = 0.5  # Irreducible error
    # generating data
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_lasso = MyLasso()
    poly_deg = [3, 5, 7, 9]
    k = 5
    lamb = np.logspace(-3.5, -2, 20)
    repeat = 30  # number of repetitions to average noise

    mse_test = np.zeros((len(poly_deg), len(lamb)))

    folds = kfold(list(range(N)), k)
    fig = plt.figure(figsize=(8, 6))

    # iterate over complexity and penalty
    for i in range(len(poly_deg)):
        for j in range(len(lamb)):
            for r in range(repeat):  # repetitions to average noise
                x = np.random.uniform(0, 1, (N, 2))
                z = frankeFunction(x[:, 0], x[:, 1]) + \
                    np.random.normal(0, sigma2, N)
                for l in range(k):
                    train_idx, test_idx = folds(l)
                    model_lasso.fit(x[train_idx], z[train_idx],
                                    poly_deg[i], lamb[j])
                    mse_test[i, j] += model_lasso.mse(x[test_idx], z[test_idx])

            mse_test[i, j] /= k * repeat

        label_str = "Model Complexity: {}".format(i)
        plt.plot(np.log10(lamb), mse_test[i], label=label_str)

    plt.grid()
    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.ylabel("Test MSE")
    plt.tight_layout(True)
    plt.legend(loc='best')
    fig.savefig(fig_path("lasso_best_model.pdf"))


def lasso_bias_variance():
    """
    Calculate the bias-variance tradeoff using MC
    """
    N = 300  # number of data points
    sigma2 = 0.5  # irreducible error
    # generating reference sample
    x = np.random.uniform(0, 1, (N, 2))
    z_noiseless = frankeFunction(x[:, 0], x[:, 1])
    z = z_noiseless + np.random.normal(0, sigma2, N)
    poly_deg = 9
    lamb = np.logspace(-3.5, -3, 15)  # penalty

    model_lasso = MyLasso()
    resamples = 30
    variance = np.zeros(len(lamb))
    bias2 = np.zeros(len(lamb))

    for i in range(len(lamb)):
        predicted = np.zeros((resamples, N))
        for j in range(resamples):
            # generating new samples for Monte Carlo simulation
            x_resample = np.random.uniform(0, 1, (N, 2))
            z_resample = frankeFunction(
                x_resample[:, 0], x_resample[:, 1]) + np.random.normal(0, sigma2, N)

            model_lasso.fit(x_resample, z_resample, poly_deg, lamb[i])
            predicted[j] = model_lasso.predict(x)

        variance[i] = np.mean(np.var(predicted, axis=0))
        bias2[i] = np.mean(np.mean((predicted - z_noiseless), axis=0)**2)

    fig = plt.figure(figsize=(8, 6))
    plt.grid()
    plt.plot(np.log10(lamb), variance)
    plt.plot(np.log10(lamb), bias2)
    plt.plot(np.log10(lamb), variance + bias2)
    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.legend(["Variance", "Bias$^2$", "Variance + Bias$^2$"])
    fig.savefig(fig_path("lasso_bias_variance.pdf"))


if __name__ == "__main__":
    lasso_shrinkage()
    lasso_model_selection()
    lasso_bias_variance()
