# -*- coding: utf-8 -*-

import numpy as np

def rand_initialize_weights(l_in, l_out):

    """
    One effective strategy for random initialization is to randomly select values for Θ(l) uniformly in the range [−ε, ε]. You should use ε = 0.12.
    2 This range of values ensures that the parameters are kept small and makes the learning more efficient.
    :param l_in:
    :param l_out:
    :return:
    """
    epsilon_init = 0.12

    return np.random.rand(l_out, l_in+1) * 2 * epsilon_init -epsilon_init