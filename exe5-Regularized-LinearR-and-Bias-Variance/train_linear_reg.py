# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as opt

from linear_reg_cost_function import linear_reg_cost_function


def train_linear_reg(X, y, l, iteration=200):

    m, n = X.shape
    initial_theta = np.zeros((n, 1))

    result = opt.minimize(fun=linear_reg_cost_function, x0=initial_theta, args=(X, y, l), method='TNC', jac=True,
                          options={'maxiter': iteration})

    return result.x
