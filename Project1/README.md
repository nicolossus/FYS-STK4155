# Project 1

Made in collaboration between [Kristian](https://github.com/KristianWold), [Tobias](https://github.com/vxkc) and [Nicolai](https://github.com/nicolossus).

This repository contains programs made for project 1 in FYS-STK4155: Applied data analysis and machine learning. In project 1, we perform regression analyses with various methods on the widely used Franke's bivariate test function and on geographical terrain data. Specifically, the regression methods OLS, Ridge and Lasso are applied with the k-fold cross-validation resampling technique.

### Main content :shipit:

- [Report: FYS_STK4155_Project_1](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/FYS_STK4155_Project_1.pdf): Project report where the theoretical background is established, the implementation of the methods explained and the results are presented, discussed and concluded upon.

- [func.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/func.py): Contains classes for the regression methods and resampling technique. In more detail:
  - The LinearModel superclass for defining the model, which is set to try a two-dimensional polynomial fit of degree p.
  - OLS subclass, which performs OLS regression.
  - Ridge subclass, which performs Ridge regression.
  - MyLasso subclass, which performs Lasso regression.
  - Helper functions, such as Franke's function, procedures for splitting the data and k-fold cross-validation, is also included in this program.

- [analysis_ols.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/analysis_ols.py): Program for analysing OLS regression on Franke's function.

- [analysis_ridge.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/analysis_ridge.py): Program for analysing Ridge regression on Franke's function.

- [analysis_lasso.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/analysis_lasso.py): Program for analysing Lasso regression on Franke's function.

- [analysis_terrain.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/analysis_terrain.py): Program for analysing OLS, Ridge and Lasso regression on the geographical terrain data [SRTM_data_Norway_1.tif](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/SRTM_data_Norway_1.tif)

- [test_func.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/test_func.py): Unit tests for the classes and methods in [func.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/func.py).

### Misc. content :moyai:

- [setup.py](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/setup.py): Setup folder structure and corresponding helper functions

- [run.sh](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/run.sh): Bash script to run tests and produce results.

- [clean.sh](https://github.com/nicolossus/FYS-STK4155/blob/master/Project1/clean.sh): Bash script to clean directory of results and caches.

- The [Figures](https://github.com/nicolossus/FYS-STK4155/tree/master/Project1/Figures) folder contains all plots produced in this project.

- The [Tables](https://github.com/nicolossus/FYS-STK4155/tree/master/Project1/Tables) folder contains tables produced in raw LaTeX format.

**Usage**

Run `bash run.sh` to reproduce all results and test the implementation.
