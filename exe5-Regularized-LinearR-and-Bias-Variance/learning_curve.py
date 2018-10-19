# -*- coding: utf-8 -*-

import numpy as np

from train_linear_reg import train_linear_reg
from linear_reg_cost_function import linear_reg_cost_function

def learning_curve(X, y, X_val, y_val, l):
    m = X.shape[0]
    J_train = np.zeros(m)
    J_val = np.zeros(m)
    for i in range(1, m+1):
        X_train = X[:i, :]
        Y_train = y[:i]
        theta = train_linear_reg(X_train, Y_train, 1.0)

        #linear_reg_cost_function 方法的时候暂时没有加入x0
        J_train[i-1] = linear_reg_cost_function(theta, X_train, Y_train, 0)[0]
        J_val[i-1] = linear_reg_cost_function(theta, X_val, y_val, 0)[0]

    return (J_train, J_val)