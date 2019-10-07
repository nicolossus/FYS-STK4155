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


def OLS_stat():
    """
    Statistical summary with OLS on data of different size and varying noise.
    Summary includes training MSE, R2 and CI
    """
    N = [100, 1000]               # Number of data points
    sigma2 = [0.01, 1]            # Irreducable error

    # Initialize model
    model_ols = OLS()
    poly_deg = 5                   # complexity
    p = 0.9                        # 90% confidence interval

    # Dataframe for storing results
    df = pd.DataFrame(columns=['N', '$\sigma^2$', 'MSE', '$R^2$'])

    # Setup for plotting
    labels = generate_labels(poly_deg)
    cmap = plt.get_cmap("Greens")

    for n in N:
        for s2 in sigma2:
            x = np.random.uniform(0, 1, (n, 2))
            noise = np.random.normal(0, s2, n)
            z = frankeFunction(x[:, 0], x[:, 1]) + noise
            model_ols.fit(x, z, poly_deg)

            mse = model_ols.mse(x, z)
            r2 = model_ols.r2(x, z)
            df = df.append({'N': n, '$\\sigma^2$': s2, 'MSE': mse,
                            '$R^2$': r2}, ignore_index=True)

            CI = model_ols.confidence_interval(p)
            norm = matplotlib.colors.Normalize(vmin=-10, vmax=len(CI))

            fig = plt.figure(figsize=(8, 6))
            plt.yticks(np.arange(model_ols.params), labels)
            plt.grid()

            for i in range(len(CI)):
                plt.plot(CI[i], (i, i), color=cmap(norm(i)))
                plt.plot(CI[i], (i, i), "o", color=cmap(norm(i)))

            plt.gca().set_title("90% Confidence Interval")
            textstr = '\n'.join((
                "$N = {}$".format(n),
                "$\\sigma^2 = {}$".format(s2)))
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            plt.gca().text(0.83, 0.95, textstr, transform=plt.gca().transAxes,
                           fontsize=14,  verticalalignment='top', bbox=props)
            text_s2 = str(s2).replace(".", "_")
            fig.savefig(fig_path("conf_{}_{}.pdf".format(n, text_s2)))

    # Render dataframe to a LaTeX tabular environment table and write to file
    pd.options.display.float_format = '{:,.3f}'.format
    df = df.apply(lambda x: x.astype(
        int) if np.allclose(x, x.astype(int)) else x)
    pd.options.display.latex.escape = False
    latex = df.to_latex(index=False, column_format='cccc')
    latex = latex.replace('\\toprule', '\\hline \\hline')
    latex = latex.replace('\\midrule', '\\hline \\hline')
    latex = latex.replace('\\bottomrule', '\\hline \\hline')

    with open(tab_path('ols_stat.tex'), 'w') as f:
        f.write(latex)


def OLS_split():
    """
    Perform data split and calculate training/testing MSE
    """
    N = 300                # Number of data points
    sigma2 = 1             # Irreducable error
    ratio = 0.25           # Train/test ratio
    model_ols = OLS()      # Initialize model
    poly_deg = 7           # Polynomial degree (complexity)
    df = pd.DataFrame(columns=['N', 'sigma2',
                               'PolyDeg', 'TrainMSE', 'TestMSE'])

    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    train_idx, test_idx = split_data(list(range(N)), ratio=ratio)
    model_ols.fit(x[train_idx], z[train_idx], poly_deg)
    mse_train = model_ols.mse(x[train_idx], z[train_idx])
    mse_test = model_ols.mse(x[test_idx], z[test_idx])

    df = df.append({'N': N, 'sigma2': sigma2, 'PolyDeg': poly_deg,
                    'TrainMSE': mse_train, 'TestMSE': mse_test},
                   ignore_index=True)
    print(df)


def OLS_CV():
    """
    Calculate train/test MSE for varying complexity using CV on OLS
    """
    N = [500, 5000]
    y_lim = [[0.15, 0.6], [0.26, 0.45]]
    repeat = 25
    sigma2 = 0.5
    model_ols = OLS()
    poly_deg_max = 9
    k = 5

    mse_train = np.zeros((repeat, poly_deg_max))
    mse_test = np.zeros((repeat, poly_deg_max))

    for n, limit in zip(N, y_lim):  # calculate for small and large dataset
        for r in range(repeat):  # resample to make many models
            x = np.random.uniform(0, 1, (n, 2))
            noise = np.random.normal(0, sigma2, n)
            z = frankeFunction(x[:, 0], x[:, 1]) + noise

            for i in range(poly_deg_max):
                folds = kfold(list(range(n)), k=5)

                for j in range(k):
                    train_idx, test_idx = folds(j)
                    model_ols.fit(x[train_idx], z[train_idx], i)
                    mse_train[r,
                              i] += model_ols.mse(x[train_idx], z[train_idx])
                    mse_test[r, i] += model_ols.mse(x[test_idx], z[test_idx])

                mse_train[r, i] /= k
                mse_test[r, i] /= k

        fig = plt.figure()
        axes = plt.gca()
        axes.set_ylim(limit)
        plt.grid()

        plt.plot(np.arange(poly_deg_max), np.mean(
            mse_train, axis=0), color="blue", linewidth=3)
        plt.plot(np.arange(poly_deg_max), np.mean(
            mse_test, axis=0), color="red", linewidth=3)

        for r in range(repeat):
            plt.plot(np.arange(poly_deg_max),
                     mse_train[r], color="blue", alpha=0.1)
            plt.plot(np.arange(poly_deg_max),
                     mse_test[r], color="red", alpha=0.1)

        plt.gca().set_xlabel("Model Complexity")
        plt.gca().set_ylabel("MSE")
        plt.gca().set_title("Method: OLS w/ $k$-fold CV")
        textstr = '\n'.join((
            "$N = {}$".format(n),
            "$\\sigma^2 = {}$".format(sigma2),
            "$k = {}$".format(k)))
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        plt.gca().text(0.75, 0.95, textstr, transform=plt.gca().transAxes,
                       fontsize=14,  verticalalignment='top', bbox=props)

        plt.legend(["Training $\\overline{\\mathrm{MSE}}}$",
                    "Test $\\overline{\\mathrm{MSE}}}$"])
        fig.savefig(fig_path(f"train_test_mse_n_{n}.pdf"))


def ols_bias_variance():
    N = 1000
    sigma2 = 0.5
    x = np.random.uniform(0, 1, (N, 2))
    z_noiseless = frankeFunction(x[:, 0], x[:, 1])
    poly_deg = np.arange(1, 9)

    model_ols = OLS()
    resamples = 30
    variance = np.zeros(len(poly_deg))
    bias2 = np.zeros(len(poly_deg))

    for i in range(len(poly_deg)):
        predicted = np.zeros((resamples, N))
        for j in range(resamples):
            x_resample = np.random.uniform(0, 1, (N, 2))
            noise = np.random.normal(0, sigma2, N)
            z_resample = frankeFunction(
                x_resample[:, 0], x_resample[:, 1]) + noise

            model_ols.fit(x_resample, z_resample, poly_deg[i])
            predicted[j] = model_ols.predict(x)

        variance[i] = np.mean(np.var(predicted, axis=0))
        bias2[i] = np.mean(np.mean((predicted - z_noiseless), axis=0)**2)
    fig = plt.figure()
    plt.plot(poly_deg, variance, label="Model Variance")
    plt.plot(poly_deg, bias2, label="Model Bias")
    plt.plot(poly_deg, variance + bias2, label="Bias + Variance")
    plt.grid()
    plt.xlabel("Model Complexity")
    plt.gca().set_title("Method: OLS w/ Pseudo-Bootstrap")
    plt.legend(loc="best")
    fig.savefig(fig_path("ols_bias_variance.pdf"))


if __name__ == "__main__":
    OLS_stat()
    OLS_split()
    OLS_CV()
    ols_bias_variance()
