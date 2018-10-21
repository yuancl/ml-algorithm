# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot_data import plot_data


def visualize_boundary_linear(X, y, clf):

    plot_data(X, y)

    coef = clf.coef_.ravel()
    intercept = clf.intercept_.ravel()

    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    yp = -1.0 * (coef[0] * xp + intercept[0]) / coef[1]

    plt.plot(xp, yp, linestyle='-', color='b')
