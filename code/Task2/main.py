import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mul

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import Lasso

from preprocess import loadData
from randomForest import DecisionTree

import cProfile


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def test(args: argparse.Namespace):
    # load the data from csv file.
    print("Loading data.")
    train_x, train_y, _ = loadData(os.path.join(args.data_dir, "train.csv"), True)
    test_x, test_id = loadData(os.path.join(args.data_dir, "test.csv"))

    # Preprocessing.
    print("Preprocessing.")

    # Training and testing.
    print("Training and testing.")

    score = 0.0

    # model = Lasso(0.0001)
    model = DecisionTree(args)

    data_index = random.sample(range(train_x.shape[0]), round(train_x.shape[0] * args.rate))
    feat_index = random.sample(range(train_x.shape[1]), round(train_x.shape[1] * args.rate))
    model.fit(train_x[data_index, :][:, feat_index], train_y[data_index])
    pred_y = model.predict(train_x[data_index, :][:, feat_index])
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0

    score = f1_score(train_y[data_index], pred_y, average="macro")

    print("Train Score = %f." % score)

    pred_y = model.predict(test_x[:, feat_index])
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0

    pred_y = model.predict(test_x[:, feat_index])

    with open(os.path.join(args.output_dir, "res.csv"), "w") as fp:
        print("id,label", file=fp)
        for i in range(len(test_id)):
            print(f"{test_id[i]},{pred_y[i]}", file=fp)

    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./poly-u-comp-5434-20242-project-task-2/")

    parser.add_argument("--max_depth", type=int, default=100)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rate", type=float, default=0.5)

    args = parser.parse_args()
    args.time = time.localtime()

    args.output_dir = f"./Result/{args.time.tm_mon:02d}{args.time.tm_mday:02d}-{args.time.tm_hour:02d}{args.time.tm_min:02d}{args.time.tm_sec:02d}/"

    os.makedirs("./Result/", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    init(args)
    test(args)
    # cProfile.run("test(args)", sort="tottime")

    print("Total time = %f(s)" % (time.time() - start_time))
