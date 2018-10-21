# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def plot_data(X, y):
    pos = np.nonzero(y == 1)
    neg = np.nonzero(y == 0)
    plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
    plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')
