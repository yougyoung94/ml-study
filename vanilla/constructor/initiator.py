import numpy as np


def initiate_layer(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.1, \
           np.random.randn(output_dim, 1) * 0.1
