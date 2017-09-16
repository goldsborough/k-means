# :arrow_heading_up: :arrow_heading_up: k-means

<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

This repository contains code referenced in my blog post [Exploring k-means in Python, C++ and
CUDA], where I implement k-means in a variety of platforms. In this post I show how CUDA
implementations of k-means can outperform scikit-learn and scipy in performance by a factor of 72
and 90, respectively.

The code is not particularly tidy, but gives an idea of how to implement k-means efficiently on a GPU.

## Contents

- `python/` contains Python code for k-means using scikit-learn, scipy and a roll-it-yourself
implementation.
- `cpp/` contains C++ implementations of k-means, including one using Eigen.
- `cuda/` holds all CUDA implementations.
- `data/` has some toy data with 100 and 100k datapoints in five clusters as well as a script to
  generate more.

## Authors

[Peter Goldsborough](http://goldsborough.me) + [cat](https://goo.gl/IpUmJn)
:heart:
