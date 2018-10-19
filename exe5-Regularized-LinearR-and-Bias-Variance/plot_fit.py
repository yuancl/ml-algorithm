# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from poly_features import poly_features
from feature_normalize import feature_normalize


def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    """
    x = np.arange(min_x - 15, max_x + 25, 0.05)
    X_poly = poly_features(x, p)
    X_poly, dummy_mu, dummy_sigma = feature_normalize(X_poly, mu, sigma)
    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))
    plt.plot(x, X_poly.dot(theta), linestyle='--', marker='', color='b')