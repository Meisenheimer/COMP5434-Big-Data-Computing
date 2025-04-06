import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
from typing import Union
from memory_profiler import profile

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

from model import DecisionTree, MultilayerPerceptron


os.makedirs("./Result/", exist_ok=True)


def loadData(filename: str, feat_selection, target: bool) -> tuple:
    if (feat_selection):
        LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41"]
    else:
        LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51"]
    data = pd.read_csv(filename).replace("?", "nan")

    if (target):
        return data[LABELS].to_numpy(), data["label"].to_numpy(), data["id"].to_numpy()
    else:
        return data[LABELS].to_numpy(), data["id"].to_numpy()


def preprocess(x: np.ndarray, y: Union[np.ndarray, None], args: argparse.ArgumentParser, test: bool) -> np.ndarray:
    res_x = x
    res_y = y
    if (args.degree >= 2):
        n = res_x.shape[0]
        m = res_x.shape[1]
        tmp = np.zeros((n, m * (args.degree - 1)))
        for i in range(2, args.degree + 1):
            tmp[:, (i-2) * m:(i-1) * m] = res_x ** i
        res_x = np.concatenate((res_x, tmp), axis=1)
    res_x = (res_x - res_x.min(axis=0)) / (res_x.max(axis=0) - res_x.min(axis=0))
    res_x *= 2.0
    res_x -= 1.0
    if (test):
        return res_x
    else:
        return res_x, res_y


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return None


def getModel(args: argparse.Namespace) -> object:
    if (args.model.lower() == "decisiontree"):
        return DecisionTree(args.max_depth, args.min_samples_split, args.min_samples_leaf, args.criterion, args.device, args.n_threshold)
    elif (args.model.lower() == "mlp"):
        return MultilayerPerceptron(args)
    else:
        raise


def train(args: argparse.Namespace) -> None:
    # load the data from csv file.
    print("Loading data.")
    x, y, _ = loadData(os.path.join(args.data_dir, "train.csv"), args.feat_selection, True)

    # Preprocessing
    print("Preprocessing")
    imputer = KNNImputer(n_neighbors=5).fit(x)
    x = imputer.transform(x)
    x, y = preprocess(x, y, args, False)

    # Training and testing.
    print("Training and testing.")
    score = []
    for seed in range(args.epoch):
        # Train and test the data with different splitting, and then take the average as the result.
        print(f"Epoch {seed}.")
        args.seed = seed
        init(args)

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=args.test_size)

        model = getModel(args)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)

        score.append(f1_score(test_y, pred_y, average="macro"))

    print("Min Score = %f (%d), Mean Score = %f, Max Score = %f (%d), Var Score = %f." % (min(score), score.index(min(score)), np.mean(score), max(score), score.index(max(score)), np.var(score)), file=args.log)
    print("Min Score = %f (%d), Mean Score = %f, Max Score = %f (%d), Var Score = %f." % (min(score), score.index(min(score)), np.mean(score), max(score), score.index(max(score)), np.var(score)))

    with open(os.path.join(args.output_dir, "Score.txt"), "w", encoding="UTF-8") as fp:
        for i in range(len(score)):
            print(score[i], file=fp)
    return None


@profile(stream=open("./Result/mem.txt", "a", encoding="UTF-8"))
def test(args: argparse.Namespace):
    # load the data from csv file.
    print("Loading data.")
    train_x, train_y, _ = loadData(os.path.join(args.data_dir, "train.csv"), args.feat_selection, True)
    test_x, test_id = loadData(os.path.join(args.data_dir, "test.csv"), args.feat_selection, False)

    # Preprocessing
    print("Preprocessing")
    imputer = KNNImputer(n_neighbors=5).fit(train_x)
    train_x = imputer.transform(train_x)
    test_x = imputer.transform(test_x)
    train_x, train_y = preprocess(train_x, train_y, args, False)
    test_x = preprocess(test_x, np.zeros_like((test_x.shape[0])), args, True)

    # Training and testing.
    print("Training and testing.")
    score = 0.0

    model = getModel(args)
    model.fit(train_x, train_y)
    pred_y = model.predict(train_x)

    score = f1_score(train_y, pred_y, average="macro")

    print("Train Score = %f." % score, file=args.log)
    print("Train Score = %f." % score)

    pred_y = model.predict(test_x)

    with open(os.path.join(args.output_dir, "res.csv"), "w") as fp:
        print("id,label", file=fp)
        for i in range(len(test_id)):
            print(f"{test_id[i]},{int(pred_y[i])}", file=fp)

    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="./poly-u-comp-5434-20242-project-task-2/")

    parser.add_argument("--n_estimator", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--criterion", type=str, default="gini")
    parser.add_argument("--n_threshold", type=int, default=128)
    parser.add_argument("--split_data_size", type=float, default=0.5)

    # MLP
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--mlp_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--feat_selection", type=bool, default=False)
    parser.add_argument("--degree", type=int, default=1)

    parser.add_argument("--mode", type=str, default="All")
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.time = time.localtime()

    args.output_dir = f"./Result/{args.mode}_{args.model}_{args.time.tm_mon:02d}{args.time.tm_mday:02d}-{args.time.tm_hour:02d}{args.time.tm_min:02d}{args.time.tm_sec:02d}/"

    if (not torch.cuda.is_available() or args.device == "cpu"):
        print("No CUDA, using CPU.")
        args.device = "cpu"
    else:
        print("Using CUDA.")

    os.makedirs(args.output_dir, exist_ok=True)

    args.log = open(os.path.join(args.output_dir, "log.txt"), "w", encoding="UTF-8")

    print(args, file=args.log)

    if (args.mode.lower() == "train"):
        init(args)
        train(args)
    elif (args.mode.lower() == "test"):
        init(args)
        test(args)
    else:
        init(args)
        train(args)
        init(args)
        test(args)

    print("Total time = %f(s)" % (time.time() - start_time), file=args.log)
    print("Total time = %f(s)" % (time.time() - start_time))

    args.log.close()
