# python

You'll need to `pip install -r requirements.txt`. Then, `k_means.py` accepts the following command
line options:

```
usage: k_means.py [-h] -m {scipy,sklearn,custom} -d DATA -k CLUSTERS [-s]
                  [-i ITERATIONS] [-r RUNS]

optional arguments:
  -h, --help            show this help message and exit
  -m {scipy,sklearn,custom}, --method {scipy,sklearn,custom}
  -d DATA, --data DATA
  -k CLUSTERS, --clusters CLUSTERS
  -s, --show
  -i ITERATIONS, --iterations ITERATIONS
  -r RUNS, --runs RUNS
```

For example, run `python k_means.py -k 5 -d ../data/100k.csv -m scipy`.
