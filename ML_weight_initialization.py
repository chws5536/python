import numpy as np


def xavier_uniform_np(n_in, n_out):
    # uniform(min, max)
    return np.random.uniform(-np.sqrt(6 / (n_in + n_out)), np.sqrt(6 / (n_in + n_out)))


def xavier_normal_np(n_in, n_out):
    # normal(mu, sigma)
    return np.random.normal(0, np.sqrt(2 / (n_in + n_out)))


def he_uniform_np(n_in):
    # uniform(min, max)
    return np.random.uniform(-np.sqrt(6 / n_in), np.sqrt(6 / n_in))


def he_normal_np(n_in):
    # normal(mu, sigma)
    return np.random.normal(0, np.sqrt(2 / n_in))
