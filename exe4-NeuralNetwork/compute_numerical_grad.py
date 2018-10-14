# -*- coding: utf-8 -*-

import numpy as np
from nn_cost_fun import nn_cost_fun

def compute_numerical_grad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda, h=0.00001):
    params_cnt = nn_params.shape[0]
    param_gradient = np.zeros(params_cnt)
    for i in range(params_cnt) :
        nn_params_temp = np.copy(nn_params)
        nn_params_temp[i] = nn_params[i] + h
        j_1 = nn_cost_fun(nn_params_temp, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)[0]
        nn_params_temp = np.copy(nn_params)
        nn_params_temp[i] = nn_params[i] - h
        j_2 = nn_cost_fun(nn_params_temp, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda)[0]

        param_gradient[i] = (j_1 - j_2)/(2*h)

    return param_gradient