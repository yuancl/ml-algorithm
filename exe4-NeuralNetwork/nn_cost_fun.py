# -*- coding: utf-8 -*-

import numpy as np
from sigmoid import sigmoid
from sigmoid_grad import sigmoid_grad

def nn_cost_fun(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
    (m, n) = X.shape
    # theta1 25*401
    # theta2 10*26
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

    Y = np.zeros([m, num_labels])
    for i in range(m):
        Y[i, y[i]] = 1
        # Y[i, y[i]-1] = 1

    # 前向计算
    # a_1 5000*401
    a_1 = np.c_[np.ones(m), X]

    # z_2 5000*25
    z_2 = a_1.dot(Theta1.T)
    # a_2 5000*26
    a_2 = np.c_[np.ones(m), sigmoid(z_2)]

    # z_3 5000*10
    z_3 = a_2.dot(Theta2.T)
    a_3 = sigmoid(z_3)

    # 注意正则化项并没有计算bias项
    J = 1.0 / m * np.sum(-Y * np.log(a_3) - (1 - Y) * (np.log(1 - a_3)))
    J_reg = J + (reg_lambda / (2.0 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    #     J = -1*(1/m)*np.sum((np.log(a3.T)*(y_matrix)+np.log(1-a3).T*(1-y_matrix))) + \
    #         (reg/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))


    # 后向计算,对z求偏导数,记为delta
    # delta_3 5000*10
    delta_3 = a_3 - Y

    # delta_2 5000*25
    delta_2 = delta_3.dot(Theta2[:, 1:]) * sigmoid_grad(z_2)

    # 后向计算,求参数的梯度,梯度的维度和thate一致
    # 首先初始化梯度
    # 10*26,25*401
    d_2 = np.zeros(Theta2.shape)
    d_1 = np.zeros(Theta1.shape)
    # Δ(2)=Δ(2)+a(2)∗δ(3) 10*26
    d_2 = d_2 + delta_3.T.dot(a_2)  # d_2初始化为0，可以不用初始化
    # Δ(1)=Δ(1)+a(1)∗δ(2) 25*401
    d_1 = d_1 + delta_2.T.dot(a_1)

    Theta1_grad = d_1 / m + 1.0*reg_lambda/m*Theta1
    Theta2_grad = d_2 / m + 1.0*reg_lambda/m*Theta2

    theta_grad = np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()]
    # print 'J = ',J_reg
    return (J_reg, theta_grad)