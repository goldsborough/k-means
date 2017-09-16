import argparse
import time

import numpy as np
import sklearn.cluster
import scipy.cluster.vq


def k_means(data, k, number_of_iterations):
    n = len(data)
    number_of_features = data.shape[1]
    # Pick random indices for the initial centroids.
    initial_indices = np.random.choice(range(n), k)
    # We keep the centroids as |features| x k matrix.
    means = data[initial_indices].T
    # To avoid loops, we repeat the data k times depthwise and compute the
    # distance from each point to each centroid in one step in a
    # n x |features| x k tensor.
    repeated_data = np.stack([data] * k, axis=-1)
    all_rows = np.arange(n)
    zero = np.zeros([1, 1, 2])
    for _ in range(number_of_iterations):
        # Broadcast means across the repeated data matrix, gives us a
        # n x k matrix of distances.
        distances = np.sum(np.square(repeated_data - means), axis=1)
        # Find the index of the smallest distance (closest cluster) for each
        # point.
        assignment = np.argmin(distances, axis=-1)
        # Again to avoid a loop, we'll create a sparse matrix with k slots for
        # each point and fill exactly the one slot that the point was assigned
        # to. Then we reduce across all points to give us the sum of points for
        # each cluster.
        sparse = np.zeros([n, k, number_of_features])
        sparse[all_rows, assignment] = data
        # To compute the correct mean, we need to know how many points are
        # assigned to each cluster (without a loop).
        counts = (sparse != zero).sum(axis=0)
        # Compute new assignments.
        means = sparse.sum(axis=0).T / counts.clip(min=1).T
    return means.T


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--method', choices=['scipy', 'sklearn', 'custom'], required=True)
parser.add_argument('-d', '--data', required=True)
parser.add_argument('-k', '--clusters', type=int, required=True)
parser.add_argument('-s', '--show', action='store_true')
parser.add_argument('-i', '--iterations', type=int, default=300)
parser.add_argument('-r', '--runs', type=int, default=10)
options = parser.parse_args()

input_values = np.loadtxt(options.data)
data, labels = input_values[:, [0, 1]], input_values[:, 2]

total_elapsed = 0
for _ in range(options.runs):
    start = time.time()
    if options.method == 'custom':
        means = k_means(data, options.clusters, options.iterations)
    elif options.method == 'sklearn':
        kmeans = sklearn.cluster.KMeans(
            options.clusters,
            max_iter=options.iterations,
            init='random',
            n_jobs=1,
            algorithm='full',
            tol=0,
            n_init=1)
        kmeans.fit(data)
        means = kmeans.cluster_centers_
    else:
        means, _ = scipy.cluster.vq.kmeans2(
            data, options.clusters, iter=options.iterations, minit='points')
    total_elapsed += time.time() - start
print('Took {0:.8f}s ({1} runs)'.format(total_elapsed / options.runs, options.runs))

if options.show:
    import matplotlib.pyplot as plot
    plot.scatter(data[:, 0], data[:, 1], c=labels)
    plot.scatter(means[:, 0], means[:, 1], linewidths=2)
    plot.show()
