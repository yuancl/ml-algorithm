# -*- coding: utf-8 -*-

import numpy as np
import sys

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    centroids : ndarray, shape (K, n_features)
        The current centroids, where K is the number of centroids.

    Returns
    -------
    idx : ndarray, shape (n_samples, 1)
        Centroid assignments. idx[i] contains the index of the centroid closest to sample i.
    """
    m = X.shape[0]
    ids = np.zeros(m)
    distance = np.ones(m) * sys.maxint
    K = centroids.shape[0]
    for i in range(K):
        k = centroids[i]
        for m_idx in range(m):
            x = X[m_idx]
            dis = np.square(x[0] - k[0]) + np.square(x[1] - k[1])
            if dis < distance[m_idx]:
                ids[m_idx] = i
                distance[m_idx] = dis

    return ids