# -*- coding: utf-8 -*-

import numpy as np

def poly_features(X, p):
    """
    Maps X (1D vector) into the p-th power.
    """
    X_poly = np.zeros((len(X), p))

    for i in range(p):
        X_poly[:, i] = np.power(X, i + 1).ravel()

    return X_poly