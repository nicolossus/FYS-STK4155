from func import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy import stats


#generate data
np.random.seed(1)
rd.seed(1)
N = int(1e3)             #Number of data points
sigma2 = 1               #Irreducable error
x = np.random.uniform(0, 1, (N,2))
z = frankeFunction(x[:,0], x[:,1]) + np.random.normal(0, sigma2, N)


k = 5
folds = kfold(N,5)

max_poly_deg = 12
mse_train = np.zeros(max_poly_deg)
mse_test = np.zeros(max_poly_deg)

model = LinearModel()
model.ridge(x, z, 5, 0)
#model.ols(x, z, 5)
"""
for i in range(max_poly_deg):
    for j in range(k):
        train_idx, test_idx = folds(j)

        model.ridge(x[train_idx],z[train_idx], i, 0.1)
        mse_train[i] += model.mse(x[train_idx], z[train_idx])
        mse_test[i] += model.mse(x[test_idx], z[test_idx])

    mse_train[i] /= k
    mse_test[i] /= k

plt.plot(list(range(max_poly_deg)), mse_train)
plt.plot(list(range(max_poly_deg)), mse_test)
plt.show()
"""

"""
X = designMatrix(x, poly_deg)

b = np.linalg.inv(X.T @ X) @ X.T @ z

mse_1 = mse(z, X @ b)
r2_1 = r2(z, X @ b)

b_var = np.linalg.inv(X.T @ X) * N/(N-P) * mse_1

t = stats.t(df = N-P).ppf(0.95)

cinterval =  [[b[i] - b_var[i][i]*t, b[i] + b_var[i][i]*t] for i in range(P)]

train_idx, test_idx = split_data(N, p = 0.25)
X_train = designMatrix(x[train_idx], poly_deg)
X_test = designMatrix(x[test_idx], poly_deg)

b_train = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z[train_idx]

mse_train = mse(z[train_idx], X_train @ b_train)
r2_train = r2(z[train_idx], X_train @ b_train)

mse_test = mse(z[test_idx], X_test @ b_train)
r2_test = r2(z[test_idx], X_test @ b_train)

print(f"train mse is {mse_train}")
print(f"test mse is {mse_test}")

k = 5
folds = kfold(N, k)

plt.show()
"""
"""
M = 40
x_lin = np.linspace(0, 1, M)
y_lin = np.linspace(0, 1, M)

x_grid, y_grid = np.meshgrid(x_lin, y_lin)
x_lin, y_lin = np.ravel(x_grid), np.ravel(y_grid)

x = np.array([[i,j] for i,j in zip(x_lin,y_lin)])

z_lin = model.predict(x)

z_grid = np.reshape(z_lin,(M,M))
print(z_grid[20,20])

fig = plt.figure()
ax = fig.gca(projection="3d")
#ax.scatter(x, y, z, color = "k", linewidths = 0.1, edgecolors = None)
surf = ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
# Add a color bar which maps values to colors.

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
