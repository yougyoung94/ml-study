import numpy as np


def initiate_layer(input_dim, output_dim):
    return np.random.rand(input_dim, output_dim), np.random.rand(1, output_dim)
