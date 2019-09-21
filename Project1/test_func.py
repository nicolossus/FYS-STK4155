import pytest
from func import *
import numpy as np

def test_design_matrix():
    model = LinearModel()
    x = np.array([[1,10], [2,20], [3,30]])
    X_computed, P = model.design_matrix(x,2)
    X_expected = np.array([[1, 1, 1, 10, 10, 100],\
                           [1, 2, 4, 20, 40, 400],\
                           [1, 3, 9, 30, 90, 900]])
    m,n = X_expected.shape
    for i in range(m):
        for j in range(n):
            assert abs(X_computed[i,j] - X_expected[i,j]) < 1e-8

    X_computed, P = model.design_matrix(x,2,intercept = False)
    X_expected = np.array([[1, 1, 10, 10, 100],\
                           [2, 4, 20, 40, 400],\
                           [3, 9, 30, 90, 900]])
    m,n = X_expected.shape
    for i in range(m):
        for j in range(n):
            assert abs(X_computed[i,j] - X_expected[i,j]) < 1e-8

def test_ols():
    np.random.seed(1)
    model = OLS()
    N = 1000
    x = np.random.uniform(0, 1, (N,2))
    b_exp = [1, 2, 3, 4, 5, 6]
    y = b_exp[0] + b_exp[1]*x[:,0] + b_exp[2]*x[:,0]**2 + b_exp[3]*x[:,1] +\
        b_exp[4]*x[:,0]*x[:,1] + b_exp[5]*x[:,1]**2

    model.fit(x,y,2)
    b_comp = model.b

    for i in range(len(b_exp)):
        assert abs(b_comp[i] - b_exp[i]) < 1e-8



def test_ridge():
    np.random.seed(1)
    model_ols = OLS()
    model_ridge = Ridge()
    N = 100
    x = np.random.uniform(0, 1, (N,2))

    b = [1, 2, 3, 4, 5, 6]
    y = b[0] + b[1]*x[:,0] + b[2]*x[:,0]**2 + b[3]*x[:,1] \
        + b[4]*x[:,0]*x[:,1] + b[5]*x[:,1]**2


    model_ols.fit(x,y,2)
    y_exp = model_ols.predict(x)

    model_ridge.fit(x,y,2,0)
    y_comp = model_ridge.predict(x)

    for i in range(N):
        assert abs(y_exp[i] - y_comp[i]) < 1e-4
