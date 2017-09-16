import sklearn.datasets
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--clusters', type=int, required=True)
parser.add_argument('-n', '--data', type=int, default=100)
parser.add_argument('-o', '--out', default='data.csv')
parser.add_argument('-d', '--delimiter', default=' ')
options = parser.parse_args()

data, labels = sklearn.datasets.make_blobs(
    n_samples=options.data, n_features=2, centers=options.clusters)

full = np.concatenate([data, labels.reshape(-1, 1)], axis=1)
np.savetxt(
    options.out, full, fmt=['%.8f', '%.8f', '%d'], delimiter=options.delimiter)
