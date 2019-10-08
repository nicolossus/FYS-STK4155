#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random as rd

import numpy as np
import sklearn.linear_model as sk

import pytest
from func import *


def test_design_matrix():
    """
    Check that the design matrix produces the correct result
    """
    model = LinearModel()
    x = np.array([[1, 10], [2, 20], [3, 30]])
    X_computed, P = model.design_matrix(x, 2)
    X_expected = np.array([[1, 1, 10, 1, 10, 100],
                           [1, 2, 20, 4, 40, 400],
                           [1, 3, 30, 9, 90, 900]])
    m, n = X_expected.shape
    for i in range(m):
        for j in range(n):
            assert abs(X_computed[i, j] - X_expected[i, j]) < 1e-8

    X_computed, P = model.design_matrix(x, 2, intercept=False)
    X_expected = np.array([[1, 10, 1, 10, 100],
                           [2, 20, 4, 40, 400],
                           [3, 30, 9, 90, 900]])
    m, n = X_expected.shape
    for i in range(m):
        for j in range(n):
            assert abs(X_computed[i, j] - X_expected[i, j]) < 1e-8


def test_ols():
    """
    Check that ols sufficiently approximates a noiseless function
    """
    np.random.seed(1)
    model = OLS()
    N = 1000
    x = np.random.uniform(0, 1, (N, 2))
    b_exp = [1, 2, 3, 4, 5, 6]
    y = b_exp[0] + b_exp[1] * x[:, 0] + b_exp[2] * x[:, 0]**2 + b_exp[3] * x[:, 1] +\
        b_exp[4] * x[:, 0] * x[:, 1] + b_exp[5] * x[:, 1]**2

    model.fit(x, y, 2)
    y_pred = model.predict(x)

    for i in range(N):
        assert abs(y_pred[i] - y[i]) < 1e-8


def test_ridge_vs_ols():
    """
    Test that our Ridge coincides with OLS for lamda = 0
    """
    np.random.seed(1)
    model_ols = OLS()
    model_ridge = Ridge()
    N = 100
    x = np.random.uniform(0, 1, (N, 2))

    b = [1, 2, 3, 4, 5, 6]
    y = b[0] + b[1] * x[:, 0] + b[2] * x[:, 0]**2 + b[3] * x[:, 1] \
        + b[4] * x[:, 0] * x[:, 1] + b[5] * x[:, 1]**2

    model_ols.fit(x, y, 2)
    y_exp = model_ols.predict(x)

    model_ridge.fit(x, y, 2, 0)
    y_comp = model_ridge.predict(x)

    for i in range(N):
        assert abs(y_exp[i] - y_comp[i]) < 1e-8

    assert abs(model_ridge.eff_params - model_ols.eff_params) < 1e-8


def test_lasso_vs_ols():
    """
    Test that our Ridge coincides with OLS for lamda ~ 0
    """
    np.random.seed(1)
    model_ols = OLS()
    model_lasso = MyLasso()
    N = 100
    x = np.random.uniform(0, 1, (N, 2))

    b = [1, 2, 3, 4, 5, 6]
    y = b[0] + b[1] * x[:, 0] + b[2] * x[:, 0]**2 + b[3] * x[:, 1] \
        + b[4] * x[:, 0] * x[:, 1] + b[5] * x[:, 1]**2

    model_ols.fit(x, y, 2)
    y_exp = model_ols.predict(x)

    model_lasso.fit(x, y, 2, 1e-10)
    y_comp = model_lasso.predict(x)

    for i in range(N):
        assert abs(y_exp[i] - y_comp[i]) < 1e-2


def test_ridge_inf_penalty():
    """
    Check that the effective degrees of freedom approaches 1 for infinite
    penalty
    """
    np.random.seed(1)
    model_ridge = Ridge()
    N = 100
    x = np.random.uniform(0, 1, (N, 2))

    b = [1, 2, 3, 4, 5, 6]
    y = b[0] + b[1] * x[:, 0] + b[2] * x[:, 0]**2 + b[3] * x[:, 1] \
        + b[4] * x[:, 0] * x[:, 1] + b[5] * x[:, 1]**2

    model_ridge.fit(x, y, 2, 10000000)

    assert abs(model_ridge.eff_params - 1) < 1e-4


def test_ridge_vs_sklearn():
    """
    Test our Ridge regression vs sklearn Ridge regression for different lambda
    """
    np.random.seed(1)
    model = LinearModel()
    model_ridge = Ridge()
    N = 100
    x = np.random.uniform(0, 1, (N, 2))

    b = [1, 2, 3, 4, 5, 6]
    y = b[0] + b[1] * x[:, 0] + b[2] * x[:, 0]**2 + b[3] * x[:, 1] \
        + b[4] * x[:, 0] * x[:, 1] + b[5] * x[:, 1]**2

    lamb = [0, 1, 2, 3, 4]
    for i in range(len(lamb)):
        # our model
        model_ridge.fit(x, y, 2, lamb[i])

        # sklearn's Ridge
        X, P = model.design_matrix(x, 2, intercept=False)
        clf = sk.Ridge(lamb[i], fit_intercept=False)
        b_sk = [np.mean(y)]

        clf.fit(X, y)
        b_sk += clf.coef_
        for b_our_ridge, b_sk_ridge in zip(model_ridge.b, b_sk):
            assert (b_our_ridge - b_sk_ridge) < 1e-10


def test_split_data():
    """
    Check that the split function works
    """
    rd.seed(1)
    N = 1000

    train_idx, test_idx = split_data(list(range(N)))
    assert abs(len(test_idx) / (len(train_idx) + len(test_idx)) - 0.25) < 0.01

    # makes sure no elements were lost, or duplicated
    all_idx = train_idx + test_idx
    all_idx.sort()
    assert list(range(N)) == all_idx


def test_kfold():
    """
    Check that the kfold function works
    """
    N = 1000
    k = 5
    folds = kfold(list(range(N)), k)

    all_test_idx = []

    for i in range(k):
        # makes sure no elements were lost, or duplicated
        train_idx, test_idx = folds(i)
        all_test_idx += test_idx

        all_idx = train_idx + test_idx
        all_idx.sort()
        assert list(range(N)) == all_idx

    all_test_idx.sort()
    assert list(range(N)) == all_test_idx
