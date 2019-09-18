import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1)
n = 100

x = np.random.rand(n, 1)

y = 5 * x * x + np.random.randn(n, 1)

X = x**2

b = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(f"numpy: {b}")

x_lin = np.linspace(0, 1, 100)

plt.plot(x, y, "o")
plt.plot(x_lin, b[0] * x_lin**2, label="Numpy")
plt.legend(loc='best')
plt.show()


# 2)
X_lin = x_lin[:, np.newaxis]**2
linreg = LinearRegression(fit_intercept=False)
coeff = linreg.fit(X, y)

print(f"scikit: {linreg.coef_}")

ypredict = linreg.predict(X_lin)

plt.plot(x, y, "o")
plt.plot(x_lin, ypredict, label="Sci-kit")
plt.legend(loc='best')
plt.show()

# 3)
print(f"Mean squared error: {mean_squared_error(y, linreg.predict(X))}")
print(f"R2 score: {r2_score(y, linreg.predict(X))}")
