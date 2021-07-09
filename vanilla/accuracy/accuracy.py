import numpy as np


def get_accuracy(Y_hat, Y):
    return np.sum(Y_hat == Y) / Y.size
