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

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
plt.rcParams.update(params)


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
    Input name of table to load or save with extension as dtype str
    """
    return os.path.join(TABLE_PATH + "/", tab_id)
