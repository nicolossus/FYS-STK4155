# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from func import *
from setup import *

# Set seed
np.random.seed(41)
rd.seed(41)


def ridge_shrinkage():
    """
    Calculate and plot the parameters for Ridge for various penalties
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

    cmap = plt.get_cmap("nipy_spectral_r")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=model_ridge.params - 1)

    fig = plt.figure(figsize=(8, 6))
    plt.grid()
    for i in range(model_ridge.params - 1):
        plt.plot(np.log10(lamb), parameters[:, i], color=cmap(norm(i)))

    plt.plot((np.log10(lamb[0]), np.log10(lamb[-1])),
             (0, 0), color="black", ls='--', lw=2)
    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.gca().set_ylabel("Coefficients $\\beta_j$ ")
    plt.gca().set_title("Method: Ridge w/o ResamplLassoing")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=0, vmax=model_ridge.params - 1))
    plt.colorbar(sm)
    fig.savefig(fig_path("ridge_shrinkage.pdf"))


def ridge_CI():
    """
    Statistical summary with OLS on data of different size and varying noise.
    Summary includes training MSE, R2 and CI
    """
    N = 100              # Number of data points
    sigma2 = 0.01        # Irreducable error
    lamb1 = 0  # Penalty
    lamb2 = 0.01

    # Initialize model
    model_ridge = Ridge()
    poly_deg = 5                   # complexity
    p = 0.9                        # 90% confidence interval

    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_ridge.fit(x, z, poly_deg, lamb1)
    CI_no_penalty = model_ridge.confidence_interval(p)

    model_ridge.fit(x, z, poly_deg, lamb2)
    CI_penalty = model_ridge.confidence_interval(p)

    # Setup for plotting
    labels = generate_labels(poly_deg)
    cmap = plt.get_cmap("Greens")
    norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(CI_no_penalty))

    fig = plt.figure(figsize=(8, 6))
    plt.yticks(np.arange(model_ridge.params), labels)
    plt.grid()

    for i in range(len(CI_no_penalty)):
        plt.plot(CI_no_penalty[i], (i, i), color=cmap(norm(i)))
        plt.plot(CI_no_penalty[i], (i, i), "o", color=cmap(norm(i)))

    plt.gca().set_title("90% Confidence Interval")
    textstr = '\n'.join((
        "$N = {}$".format(N),
        "$\\sigma^2 = {}$".format(sigma2), "$\\lambda = {}$".format(lamb1)))
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.gca().text(0.83, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=14,  verticalalignment='top', bbox=props)
    fig.savefig(fig_path("ridge_CI_no_penalty.pdf"))

    fig = plt.figure(figsize=(8, 6))
    plt.yticks(np.arange(model_ridge.params), labels)
    plt.grid()

    for i in range(len(CI_no_penalty)):
        plt.plot(CI_penalty[i], (i, i), color=cmap(norm(i)))
        plt.plot(CI_penalty[i], (i, i), "o", color=cmap(norm(i)))

    plt.gca().set_title("90% Confidence Interval")
    textstr = '\n'.join((
        "$N = {}$".format(N),
        "$\\sigma^2 = {}$".format(sigma2), "$\\lambda = {}$".format(lamb2)))
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.gca().text(0.83, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=14,  verticalalignment='top', bbox=props)
    fig.savefig(fig_path("ridge_CI_penalty.pdf"))


def ridge_model_selection():
    """
    Calculate the test MSE of Ridge for various complexitis and penalties
    """
    N = 500  # Number of data points
    sigma2 = 0.5  # irreducible error
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_ridge = Ridge()
    poly_deg = [3, 5, 7, 9]
    k = 5
    lamb = np.logspace(-3, 1, 20)
    mse_test = np.zeros((len(poly_deg), len(lamb)))
    folds = kfold(list(range(N)), k)

    fig = plt.figure(figsize=(8, 6))
    # iterate over complexity and penalty
    for i in range(len(poly_deg)):
        for j in range(len(lamb)):
            for l in range(k):
                train_idx, test_idx = folds(l)
                model_ridge.fit(x[train_idx], z[train_idx],
                                poly_deg[i], lamb[j])
                mse_test[i, j] += model_ridge.mse(x[test_idx], z[test_idx])

            mse_test[i, j] /= k

        label_str = "Model Complexity: {}".format(i)
        plt.plot(np.log10(lamb), mse_test[i], label=label_str)

    plt.grid()
    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.ylabel("Test MSE")
    plt.tight_layout(True)
    plt.legend(loc='best')
    fig.savefig(fig_path("ridge_best_model.pdf"))


def ridge_bias_variance():
    """
    Calculate the bias-variance tradeoff using MC
    """
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
        for j in range(resamples):  # repetitions to average noise
            x_resample = np.random.uniform(0, 1, (N, 2))
            z_resample = frankeFunction(
                x_resample[:, 0], x_resample[:, 1]) + np.random.normal(0, sigma2, N)

            model_ridge.fit(x_resample, z_resample, poly_deg, lamb[i])
            predicted[j] = model_ridge.predict(x)

        variance[i] = np.mean(np.var(predicted, axis=0))
        bias2[i] = np.mean(np.mean((predicted - z_noiseless), axis=0)**2)
    fig = plt.figure(figsize=(8, 6))
    plt.grid()
    plt.plot(np.log10(lamb), variance)
    plt.plot(np.log10(lamb), bias2)
    plt.plot(np.log10(lamb), variance + bias2)
    plt.gca().set_xlabel("$\\log_{10}(\\lambda)$")
    plt.legend(["Variance", "Bias$^2$", "Variance + Bias$^2$"])
    fig.savefig(fig_path("ridge_bias_variance.pdf"))


if __name__ == "__main__":
    ridge_shrinkage()
    ridge_model_selection()
    ridge_bias_variance()
    ridge_CI()
