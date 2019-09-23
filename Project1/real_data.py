import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from func import *
from imageio import imread

# Load the terrain
terrain1 = imread("SRTM_data_Norway_1.tif")
# Show the terrain
plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain1, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

terrain_downsample = down_sample(terrain1, 10)

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_downsample, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


m, n = terrain_downsample.shape
z = np.ravel(terrain_downsample)
x = np.array([j for i in range(m) for j in n * [i]]) / m
y = np.array(m * list(range(n))) / n


print(len(z))
print(len(x))
print(len(y))
x = np.array([[i, j] for i, j in zip(x, y)])

model_ridge = Ridge()
model_ridge.fit(x, z, 50, lamb=0.00000001)

terrain_fitted = model_ridge.predict(x)
terrain_fitted = np.reshape(terrain_fitted, (m, n))

plt.figure()
plt.title("Terrain over Norway 1")
plt.imshow(terrain_fitted, cmap="gray")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
