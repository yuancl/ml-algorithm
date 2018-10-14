# -*- coding: utf-8 -*-

import numpy as np

from sigmoid import sigmoid


def predict(Theta_1, Theta_2, X):
    """
    Predict the label of an input given a trained neural network.
    Parameters
    ----------
    Theta_1 : ndarray
        Trained weights of layer 1 of the neural network.
    Theta_2 : ndarray
        Trained weights of layer 2 of the neural network.
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    Returns
    -------
    p : ndarray, shape (n_samples,)
         The prediction for x.
    """

    # m, n = X.shape
    # X = np.hstack((np.ones((m, 1)), X))
    # A_2 = sigmoid(X.dot(Theta_1.T))
    # A_2 = np.hstack((np.ones((m, 1)), A_2))
    # A_3 = sigmoid(A_2.dot(Theta_2.T))
    #
    # p = np.argmax(A_3, axis=1)
    # p += 1  # The theta_1 and theta_2 are loaded from Matlab data, in which the matrix index starts from 1.

    m, n = X.shape
    # 前向计算
    # a_1 5000*401
    a_1 = np.c_[np.ones(m), X]

    # z_2 5000*25
    z_2 = a_1.dot(Theta_1.T)
    # a_2 5000*26
    a_2 = np.c_[np.ones(m), sigmoid(z_2)]

    # z_3 5000*10
    z_3 = a_2.dot(Theta_2.T)
    a_3 = sigmoid(z_3)

    return np.argmax(a_3, axis=1)