import csv

from constructor import *


architecture = [
    {
        'input_dim': [],
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

    with open('./data/train.csv', 'rb') as file:
        train = csv.reader(file)

    pass
