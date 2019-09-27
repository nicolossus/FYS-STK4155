#!/usr/bin/env python3
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

"""
DEBUGGER
remove when done
"""
# sys.exit(0)

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)

# Set seed
np.random.seed(42)
rd.seed(42)

# Set path to save the figures and data files
ROOT = str(os.getcwd())
PROJECT = ROOT
PROJECT_ID = "/Project1"
FIGURE_ID = "/Figures"
TABLE_ID = "/Tables"

if PROJECT_ID not in PROJECT:
    PROJECT += PROJECT_ID

if not os.path.exists(PROJECT + FIGURE_ID):
    os.makedirs(PROJECT + FIGURE_ID)

if not os.path.exists(PROJECT + TABLE_ID):
    os.makedirs(PROJECT + TABLE_ID)

FIGURE_PATH = PROJECT + FIGURE_ID
TABLE_PATH = PROJECT + TABLE_ID


def fig_path(fig_id):
    """
    Input name of figure to load or save with extension as dtype str
    """
    return os.path.join(FIGURE_PATH + "/", fig_id)


def tab_path(tab_id):
    """
    Input name of figure to load or save with extension as dtype str
    """
    return os.path.join(TABLE_PATH + "/", tab_id)


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
            z = frankeFunction(x[:, 0], x[:, 1] + noise)
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

            fig.suptitle("90% Confidence Interval")
            textstr = '\n'.join((
                "$N = {}$".format(n),
                "$\\sigma^2 = {}$".format(s2)))
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            plt.gca().text(0.83, 0.95, textstr, transform=plt.gca().transAxes,
                           fontsize=14,  verticalalignment='top', bbox=props)
            fig.savefig(fig_path("conf_{}_{}.pdf".format(n, s2)))

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
    Perform data split and calculate training/testing mse
    """
    N = 300                # Number of data points
    sigma2 = 1             # Irreducable error
    ratio = 0.25           # Train/test ratio
    model_ols = OLS()      # Initialize model
    poly_deg = 7           # Polynomial degree (complexity)

    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    train_idx, test_idx = split_data(list(range(N)), ratio=ratio)
    model_ols.fit(x[train_idx], z[train_idx], poly_deg)
    mse_train = model_ols.mse(x[train_idx], z[train_idx])
    mse_test = model_ols.mse(x[test_idx], z[test_idx])
    print(
        f"mse_train = {mse_train:.3f} , mse_test = {mse_test:.3f}, for N = {N}"
        + f", sigma2 = {sigma2}, poly_deg = {poly_deg}")


def OLS_CV():
    """
    Calculate train/test MSE for varying complexity using CV on OLS
    """
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
                    mse_train[r,
                              i] += model_ols.mse(x[train_idx], z[train_idx])
                    mse_test[r, i] += model_ols.mse(x[test_idx], z[test_idx])

                mse_train[r, i] /= k
                mse_test[r, i] /= k

        fig = plt.figure()
        fig.suptitle(f"Train vs Test MSE, N = {N[n]}, $\\sigma^2$ = {sigma2}")
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
            plt.plot(np.arange(poly_deg_max),
                     mse_test[r], color="red", alpha=0.1)

        plt.legend(["train MSE", "test MSE"])
        fig.savefig(fig_path(f"train_test_mse_{n}_{sigma2}.pdf"))


def Ridge_model():
    """
    Ridge
    """
    N = 1000
    sigma2 = 1
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_ridge = Ridge()
    poly_deg = 5
    lamb = np.logspace(1, 5, 15)
    parameters = []

    for i in range(len(lamb)):
        model_ridge.fit(x, z, poly_deg, lamb[i])
        parameters.append(model_ridge.b[1:])
    parameters = np.array(parameters)

    cmap = plt.get_cmap("Greens")
    norm = matplotlib.colors.Normalize(vmin=-10, vmax=model_ridge.params - 1)

    plt.grid()
    for i in range(model_ridge.params - 1):
        plt.plot(np.log(lamb), parameters[:, i], color=cmap(norm(i)))

    plt.plot((np.log(lamb[0]), np.log(lamb[-1])),
             (0, 0), color="black", linewidth=2)
    plt.show()


def Lasso_model():
    """
    Lasso
    """
    N = 1000
    sigma2 = 0.01
    x = np.random.uniform(0, 1, (N, 2))
    z = frankeFunction(x[:, 0], x[:, 1]) + np.random.normal(0, sigma2, N)

    model_lasso = MyLasso()
    poly_deg = 5
    lamb = np.logspace(1e-5, 1e-2, 15)
    parameters = []

    for i in range(len(lamb)):
        model_lasso.fit(x, z, poly_deg, lamb[i])
        print(model_lasso.b[0])
        parameters.append(model_lasso.b)
    parameters = np.array(parameters)

    cmap = plt.get_cmap("Greens")
    norm = matplotlib.colors.Normalize(vmin=-10, vmax=model_lasso.params - 1)

    plt.grid()
    for i in range(model_lasso.params - 1):
        plt.plot(np.log(lamb), parameters[:, i], color=cmap(norm(i)))

    plt.plot((np.log(lamb[0]), np.log(lamb[-1])),
             (0, 0), color="black", linewidth=2)
    plt.show()


if __name__ == "__main__":
    # OLS_stat()
    OLS_split()
    # OLS_CV()
    # Ridge_model()
    # Lasso_model()
