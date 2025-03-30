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
from sklearn.ensemble import RandomForestClassifier

from preprocess import loadData


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def train(args: argparse.Namespace) -> None:
    # load the data from csv file.
    print("Loading data.")
    x, y, _ = loadData(os.path.join(args.data_dir, "train.csv"), True)
    print(x.shape)
    print(y.shape)

    # Training and testing.
    print("Training and testing.")
    score = []
    for seed in tqdm(range(args.epoch)):
        # Train and test the data with different splitting, and then take the average as the result.
        args.seed = seed
        init(args)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=args.test_size)

        model = Lasso(0.0001)
        # model = RandomForestClassifier(n_estimators=10, max_depth=100, n_jobs=mul.cpu_count())

        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        pred_y[pred_y > 0.5] = 1
        pred_y[pred_y <= 0.5] = 0

        score.append(f1_score(test_y, pred_y, average="macro"))

    print("Min Score = %f (%d), Mean Score = %f, Max Score = %f (%d), Var Score = %f." % (min(score), score.index(min(score)), np.mean(score), max(score), score.index(max(score)), np.var(score)), file=args.log)

    with open(os.path.join(args.output_dir, "Score.txt"), "w", encoding="UTF-8") as fp:
        for i in range(len(score)):
            print(score[i], file=fp)
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

    model = Lasso(0.0001)
    # model = RandomForestClassifier(n_estimators=10, max_depth=100, n_jobs=mul.cpu_count())

    model.fit(train_x, train_y)
    pred_y = model.predict(train_x)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0

    score = f1_score(train_y, pred_y, average="macro")

    print("Train Score = %f." % score, file=args.log)

    pred_y = model.predict(test_x)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0

    with open(os.path.join(args.output_dir, "res.csv"), "w") as fp:
        print("id,label", file=fp)
        for i in range(len(test_id)):
            print(f"{test_id[i]},{pred_y[i]}", file=fp)

    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)

    parser.add_argument("--mode", type=str, default="")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--test_size", type=float, default=0.2)

    args = parser.parse_args()
    args.time = time.localtime()

    args.output_dir = f"./Result/{args.time.tm_mon:02d}{args.time.tm_mday:02d}-{args.time.tm_hour:02d}{args.time.tm_min:02d}{args.time.tm_sec:02d}/"

    os.makedirs("./Result/", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    args.log = open(os.path.join(args.output_dir, "log.txt"), "w", encoding="UTF-8")

    print(args, file=args.log)

    init(args)
    if (args.mode.lower() == "train"):
        train(args)
    elif (args.mode.lower() == "test"):
        test(args)
    else:
        train(args)
        test(args)

    args.log.close()

    print("Total time = %f(s)" % (time.time() - start_time))
