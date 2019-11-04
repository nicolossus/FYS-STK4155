import numpy as np
from sklearn.datasets import load_digits

import pickle
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from urllib.request import urlopen


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def deriv(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2


class Tanh():
    def __call__(self, x):
        return np.tanh(x)

    def deriv(self, x):
        return 1 + np.tanh(x)**2


class Relu():
    def __call__(self, x):
        return np.max(x, 0)

    def deriv(self, x):
        return 0 < x


class SoftMax():
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]

    def deriv(self, x):
        return self(x) * (1 - self(x))


class Pass():
    def __call__(self, x):
        return x

    def deriv(self, x):
        return 1


class SquareLoss():
    def __call__(self, y_pred, y):
        return 0.5 * sum((y_pred - y)**2)

    def deriv(self, y_pred, y):
        return y_pred - y


class CrossEntropy():
    def __call__(self, y_pred, y):
        return -sum(y * np.log(y_pred) + (1 - y) * log(1 - y_pred))

    def deriv(self, y_pred, y):
        return (y_pred - y) / (y_pred * (1 - y_pred))


class MlogLoss():
    def __call__(self, y_pred, y):
        return -sum(y * np.log(y_pred) + (1 - y) * log(1 - y_pred))

    def deriv(self, y_pred, y):
        return -y.T / y_pred


class NeuralNetwork():

    def __init__(self, dim, acf, cost):
        self.dim = dim
        self.acf = np.array(acf)
        self.cost = cost

        self.W = np.empty(len(dim) - 1, dtype=np.ndarray)  # weight matricies
        self.b = np.empty(len(dim) - 1, dtype=np.ndarray)  # biases
        self.z = np.empty(len(dim), dtype=np.ndarray)
        self.a = np.empty(len(dim), dtype=np.ndarray)
        self.grad = np.empty(len(dim) - 1, dtype=np.ndarray)
        self.delta = np.empty(len(dim) - 1, dtype=np.ndarray)

        for i in range(len(dim) - 1):
            m = dim[i + 1]
            n = dim[i]
            self.W[i] = np.random.normal(0, 1, (m, n))
            self.b[i] = 0.01 * np.ones(m)

    def forward(self, x):
        self.z[0] = x
        self.a[0] = x
        for i in range(len(self.W)):
            self.z[i + 1] = self.a[i]@self.W[i].T + self.b[i][np.newaxis]
            self.a[i + 1] = self.acf[i](self.z[i + 1])

    def backward(self, x, y):
        self.forward(x)

        self.delta[-1] = self.acf[-1].deriv(self.z[-1]) * \
            self.cost.deriv(self.a[-1], y)

        for i in range(len(self.W) - 1, 0, -1):
            self.delta[i - 1] = self.delta[i] @ self.W[i] * \
                self.acf[i - 1].deriv(self.z[i])

    def train(self, X, y, mu, batch_size):
        if len(y.shape) == 1:
            y = y[:, np.newaxis]

        n = len(y)
        num_iters = int(n / batch_size)

        for i in range(num_iters):
            idx_train = np.random.choice(
                np.arange(0, n), batch_size, replace=False)
            self.backward(X[idx_train], y[idx_train])
            for j in range(len(self.grad)):
                self.grad[j] = self.delta[j].T @ self.a[j]

            self.W -= mu * self.grad

            for j in range(len(self.grad)):
                self.b[j] -= mu * np.sum(self.delta[j], axis=0)


tanh = Tanh()
sig = Sigmoid()
softMax = SoftMax()
relu = Relu()

crossEntropy = CrossEntropy()
squareLoss = SquareLoss()
mlogloss = MlogLoss()

"""
data = load_digits()

X = data.data
y = data.target


np.random.seed(42)
idx = np.where(np.logical_or(y == 0, y == 1))[0]
np.random.shuffle(idx)
idx_train = idx[:250]
idx_test = idx[250:]

X_train = X[idx_train]
y_train = y[idx_train]

X_test = X[idx_test]
y_test = y[idx_test]
"""

url_main = "https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/"
data_file_name = "Ising2DFM_reSample_L40_T=All.pkl"
label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"

print("1")

data = pickle.load(urlopen(url_main + data_file_name))
print("2")
data = np.unpackbits(data).reshape(-1, 1600)
print("3")
data = data.astype('int')

print(data.shape)
"""
nn = NeuralNetwork((64, 10, 1), [tanh, sig], crossEntropy)

epoch = 1000

for i in range(epoch):
    nn.train(X_train, y_train, 0.001, 64)
    if i % (epoch / 100) == 0:
        print(i * (100 / epoch))

success = 0

nn.forward(X_test)

print(y_test[:10])
print(np.round((nn.a)[-1][:10]))
# print(y_test.shape)

for i in range(len(y_test)):
    success += (np.round((nn.a)[-1][i]) == y_test[i])

print(success, "/", len(y_test))
"""
