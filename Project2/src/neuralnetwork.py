import numpy as np
import numba as nb
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ShuffleSplit


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
        return np.exp(x) / np.sum(np.exp(x), axis=0)

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
        ting = np.sum(y_pred - y.T, axis=0)
        return ting


class CrossEntropy():
    def __call__(self, y_pred, y):
        return -sum(y * np.log(y_pred) + (1 - y) * log(1 - y_pred))

    def deriv(self, y_pred, y):
        return (y_pred - y.T) / (y_pred * (1 - y_pred))


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
        self.z[0] = x.T
        self.a[0] = x.T
        for i in range(len(self.W)):
            self.z[i + 1] = self.W[i]@self.a[i] + self.b[i][:, np.newaxis]
            self.a[i + 1] = self.acf[i](self.z[i + 1])

    def backward(self, x, y):
        self.forward(x)
        self.grad[-1] = self.acf[-1].deriv(self.z[-1]) * \
            self.cost.deriv(self.a[-1], y)

        for i in range(len(self.W) - 1, 0, -1):
            self.grad[i - 1] = self.W[i].T @ self.grad[i] * \
                self.acf[i].deriv(self.z[i])

    def train(self, X, y, mu, batch_size):
        self.backward(X, y)
        for i in range(len(self.grad)):
            self.delta[i] = self.grad[i]@self.a[i].T

        self.W -= mu * self.delta
        for i in range(len(self.grad)):
            self.b[i] -= mu * np.sum(self.grad[i], axis=1)

        """
        n = len(y)
        num_iters = int(n / batch_size)


        for i in range(num_iters):
            idx_train = np.random.choice(
                np.arange(0, n), batch_size, replace=False)

            self.backward(X[idx_train], y[idx_train])

            for j in range(len(self.grad)):
                self.delta[i] = self.grad[i]@self.a[i].T

            self.W -= mu * self.delta
            for i in range(len(self.grad)):
                self.b[i] -= mu * np.sum(self.grad[i], axis=1)

        """


tanh = Tanh()
sig = Sigmoid()
softMax = SoftMax()
relu = Relu()

crossEntropy = CrossEntropy()
squareLoss = SquareLoss()
mlogloss = MlogLoss()

data = load_digits()

X = data.data
y = data.target


np.random.seed(42)
idx = np.where(np.logical_or(y == 0, y == 1))[0]
np.random.shuffle(idx)
idx_train = idx[:300]
idx_test = idx[300:]

X_train = X[idx_train]
y_train = y[idx_train]

X_test = X[idx_test]
y_test = y[idx_test]


nn = NeuralNetwork((64, 32, 1), [tanh, sig], crossEntropy)

epoch = 1000

for i in range(epoch):
    nn.train(X_train, y_train, 0.00001, 64)
    if i % (epoch / 100) == 0:
        print(i * (100 / epoch))

success = 0

nn.forward(X_test)

# print(y_test[:10])
print((nn.a)[-1].shape)
print(y_test.shape)

for i in range(len(y_test)):
    success += (np.round((nn.a)[-1][:, i]) == y_test[i])

print(success)

"""

enc = OneHotEncoder(categories='auto')

N = 1500
N_test = 100

y = enc.fit_transform(np.array(data.target).reshape(-1, 1)).toarray()
x = np.array(data.data)

np.random.seed(42)
nn = NeuralNetwork((64, 48, 10), [tanh, softMax], squareLoss)

y_train = y[:1300]
x_train = x[:1300] / np.max(x)

y_test = y[1300:]
x_test = x[1300:] / np.max(x)

epoch = 1000

for i in range(epoch):
    nn.train(x_train, y_train, 0.0001, 0, 64)
    if i % (epoch / 100) == 0:
        print(i * (100 / epoch))

success = 0

nn.forward(x_train)

for i in range(100):
    success += np.array_equal(np.round((nn.a)[-1][:, i]), y_train[i])

print(success)
"""
