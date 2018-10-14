# -*- coding: utf-8 -*-

from sigmoid import sigmoid

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))
