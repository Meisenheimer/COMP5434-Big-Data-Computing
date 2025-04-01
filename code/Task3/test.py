import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mul

from sklearn.metrics import f1_score
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation

from preprocess import loadData

MAX_ITER = 8192


def main() -> None:
    # load the data from csv file.
    print("Loading data.")
    x, y, _ = loadData(os.path.join("./poly-u-comp-5434-20242-project-task-3/train.csv"), True)
    print(x.shape)
    print(y.shape)

    # Preprocessing
    print("Preprocessing")
    index = [1, 2, 3, 6, 9, 10, 11, 13, 15, 16]
    # index = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 15, 16, 17, 26, 27]
    x = x[:, index]
    # cluster = KMeans(n_init="auto")
    cluster = DBSCAN(eps=0.5, min_samples=2, n_jobs=mul.cpu_count())
    label = cluster.fit_predict(np.concatenate((x, 10 * y.reshape(-1, 1)), axis=1))
    index = np.zeros(x.shape[0])
    s = 0
    for i in np.unique(label):
        if (len(y[label == i]) >= 1000):
            index[label == i] = 1
            s += len(y[label == i])
            p = np.count_nonzero(y[label == i]) / float(len(y[label == i]))
            print(i, len(y[label == i]), 1 - p, p)
    print(np.count_nonzero(index), s)
    print(np.unique(index))

    return None


if __name__ == "__main__":
    start_time = time.time()

    np.random.seed(42)
    random.seed(42)
    main()

    print("Total time = %f(s)" % (time.time() - start_time))
