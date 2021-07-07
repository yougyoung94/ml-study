import csv

import numpy as np

from constructor import forward_propagation, backward_propagation


architecture = [
    {
        'input_dim': [784, ],
        'output_dim': [],
        'activation': 'relu',
    },
    {
        'input_dim': [],
        'output_dim': [],
        'activation': 'relu',
    },
    {
        'input_dim': [],
        'output_dim': [],
        'activation': 'sigmoid',
    },
]


if __name__ == '__main__':
    data = np.genfromtxt('../data/train.csv', delimiter=',', skip_header=1)
    X, Y = data[:, 1:], data[:, 0]

    model = 

    pass
