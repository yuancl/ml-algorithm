# -*- coding: utf-8 -*-

import numpy as np

def linear_reg_cost_function(theta, X, y,  lambda_):
    """
    X is an two dimen array: each col is a feature,and each row is an example
    注意正则化部分是不计算theta0的
    """
    H = theta.dot(X.T)
    m  = X.shape[0]
    J = 1.0/(2*m)*np.sum(np.square(H-y)) + lambda_/(2.0*m) * np.sum(theta[1:])

    #梯度计算
    grad = np.zeros(theta.shape[0])
#     grad[0] = 1.0/m*np.sum((H - y)*1)
#     grad[1:] = (1.0/m*np.sum((H - y).dot(X[:,1:])) + lambda_/m*theta[1:])
    mask = np.eye(len(theta))
    mask[0, 0] = 0
    grad = 1.0 / m * X.T.dot(X.dot(theta) - y) + 1.0 * lambda_ / m * (mask.dot(theta))

    return (J, grad)