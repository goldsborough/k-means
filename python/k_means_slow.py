import argparse
import time

import numpy as np
import sklearn.datasets


def k_means(data, k, number_of_iterations):
    initial_indices = np.random.choice(range(len(data)), k)
    means = data[initial_indices]
    repeated_data = np.stack([data] * k, axis=-1)
    for _ in range(number_of_iterations):
        assignment = np.zeros(len(data), np.int32)
        for index in range(len(data)):
            best = np.sum(np.square(means - data[index]), axis=1).argmin()
            assignment[index] = best
        # distances = np.sum(np.square(repeated_data - means), axis=1)
        # assignment = np.argmin(distances, axis=-1)
        new_means = np.zeros_like(means)
        counts = np.zeros(k)
        for index in range(len(data)):
            new_means[assignment[index]] += data[index]
            counts[assignment[index]] += 1
        means = new_means / counts.reshape(k, 1).clip(min=1)
        assert not np.isnan(means).any()

    return means.T


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--clusters', type=int, required=True)
parser.add_argument('-d', '--display', action='store_true')
parser.add_argument('-i', '--iterations', type=int, default=300)
parser.add_argument('-n', '--data', type=int, default=100)
options = parser.parse_args()

data, labels = sklearn.datasets.make_blobs(
    n_samples=options.data, n_features=2, centers=options.clusters)

start = time.time()
means = k_means(data, options.clusters, options.iterations)
print('Took {0:.3f}s'.format(time.time() - start))

if options.display:
    import matplotlib.pyplot as plot
    plot.scatter(data[:, 0], data[:, 1], c=labels)
    plot.scatter(means[:, 0], means[:, 1], linewidths=2)
    plot.show()
