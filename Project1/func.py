import random as rd

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D


def frankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term3


def designMatrix(x, p):
    n = x.shape[0]
    P = int(((p + 2) * (p + 1)) / 2)
    X = np.zeros((n, P))
    idx = 0
    for i in range(p + 1):
        for j in range(p - i + 1):
            X[:, idx] = (x[:, 0]**i) * (x[:, 1]**j)
            idx += 1
    return X


def mse(y, y_pred):
    n = y.size
    mse = 1 / n * np.sum((y - y_pred)**2)
    return mse


def r2(y, y_pred):
    n = y.size
    y_ave = np.mean(y)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_ave)**2)
    return r2


def split_data(n, p=0.25):
    test_n = int(p * n)
    idx = list(range(n))
    rd.shuffle(idx)
    test_idx = [idx.pop() for i in range(test_n)]
    train_idx = idx
    return test_idx, train_idx


def kfold(n, k=5):
    idx = np.array(list(range(n)))
    np.random.shuffle(idx)
    idx = np.array_split(idx, k)

    def folds(i):
        test_idx = idx[i]
        train_idx = np.concatenate((idx[:i], idx[i + 1:]), axis=None)
        train_idx = train_idx.astype("int16")
        return train_idx, test_idx

    return folds


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Make data.
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)

    print(x)

    z = frankeFunction(x, y)

    # Plot the surface.

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    # Add a color bar which maps values to colors.

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
