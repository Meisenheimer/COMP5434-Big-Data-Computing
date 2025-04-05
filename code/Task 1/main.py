import os
import time
import torch
import random
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mul
from memory_profiler import profile


from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from pre_work import Engineering
from model import LogisticModel, RandomForest


def loadData(filename: str, target: bool = False) -> tuple:
    data = pd.read_csv(filename)
    if (target):
        return data.drop(['label'], axis=1).to_numpy(), data["label"].to_numpy()
    else:
        return data.drop(['StudentID'], axis=1).to_numpy(), data["StudentID"].to_numpy()


def preprocess(x: np.ndarray, args: argparse.ArgumentParser) -> np.ndarray:
    res = x
    if (args.degree >= 2):
        n = x.shape[0]
        m = x.shape[1]
        tmp = np.zeros((n, m * (args.degree - 1)))
        for i in range(2, args.degree + 1):
            tmp[:, (i-2) * m:(i-1) * m] = x ** i
        res = np.concatenate((x, tmp), axis=1)
    if (args.raising):
        n = x.shape[0]
        m = x.shape[1]
        tmp = np.zeros((n, m * (m - 1) // 2))
        k = 0
        for i in range(m):
            for j in range(i + 1, m):
                tmp[:, k] = x[:, i] * x[:, j]
                k += 1
        res = np.concatenate((x, tmp), axis=1)
    res = (res - res.min(axis=0)) / (res.max(axis=0) - res.min(axis=0))
    res *= 2.0
    res -= 1.0
    return res


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return None


def getModel(args: argparse.Namespace) -> object:
    if (args.model.lower() == "logistic"):
        # if (args.alpha_1 == 0.0 and args.alpha_2 == 0.0):
        #     return LogisticRegression(penalty=None, max_iter=args.max_iter, solver="saga")
        # else:
        #     return LogisticRegression(penalty="elasticnet", C=args.alpha_1 + args.alpha_2, max_iter=args.max_iter, solver="saga", l1_ratio=args.alpha_1 / (args.alpha_1 + args.alpha_2))
        return LogisticModel(args)

    elif (args.model.lower() == "randomforest"):
        # return RandomForestClassifier(n_estimators=args.n_estimator, max_depth=args.max_depth, min_samples_split=args.min_samples_split, criterion=args.criterion, n_jobs=mul.cpu_count())
        return RandomForest(args)
    else:
        raise


@profile
def train(args: argparse.Namespace) -> None:
    # load the data from csv file.
    print("Loading data.")
    work = Engineering()
    df = work.eda('train.csv')
    work.feature_engineering(df, target=True)
    x, y = loadData(os.path.join(args.data_dir, "train_new.csv"), True)

    # Preprocessing
    print("Preprocessing")
    x = preprocess(x, args)

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
        # print("train_x shape:", train_x.shape)
        # print("train_y shape:", train_y.shape)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)

        score.append(f1_score(test_y, pred_y, average="macro"))

    print("Min Score = %f (%d), Mean Score = %f, Max Score = %f (%d), Var Score = %f." % (min(score), score.index(min(score)), np.mean(score), max(score), score.index(max(score)), np.var(score)), file=args.log)
    print("Min Score = %f (%d), Mean Score = %f, Max Score = %f (%d), Var Score = %f." % (min(score), score.index(min(score)), np.mean(score), max(score), score.index(max(score)), np.var(score)))

    with open(os.path.join(args.output_dir, "Score.txt"), "w", encoding="UTF-8") as fp:
        for i in range(len(score)):
            print(score[i], file=fp)
    return None


def test(args: argparse.Namespace):
    # load the data from csv file.
    print("Loading data.")
    work = Engineering()
    df = pd.read_csv("test.csv")
    work.feature_engineering(df, target=False)
    train_x, train_y = loadData(os.path.join(args.data_dir, "train_new.csv"), True)
    test_x, test_id = loadData(os.path.join(args.data_dir, "test_new.csv"))

    # Preprocessing
    print("Preprocessing")
    train_x = preprocess(train_x, args)
    test_x = preprocess(test_x, args)

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
        print("StudentID,label", file=fp)
        for i in range(len(test_id)):
            print(f"{test_id[i]},{int(pred_y[i])}", file=fp)

    return None

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="")

    parser.add_argument("--n_estimator", type=int, default=1)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--n_threshold", type=int, default=128)
    parser.add_argument("--split_data_size", type=float, default=1)
    parser.add_argument("--criterion", type=str, default="entropy")

    parser.add_argument("--alpha_1", type=float, default=0.0)
    parser.add_argument("--alpha_2", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.15)
    parser.add_argument("--tol", type=float, default=1e-3)

    parser.add_argument("--raising", type=bool, default=False)  # Ture: 生成原始特征的交叉项（两两相乘）
    parser.add_argument("--degree", type=int, default=1)        # 如果大于等于2，会生成原始特征的高次项

    parser.add_argument("--mode", type=str, default="All")
    # parser.add_argument("--model", type=str, default="logistic")
    parser.add_argument("--model", type=str, default="randomforest")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--test_size", type=float, default=0.2)

    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    args.time = time.localtime()

    args.output_dir = f"result/{args.mode}_{args.model}_{args.time.tm_mon:02d}{args.time.tm_mday:02d}-{args.time.tm_hour:02d}{args.time.tm_min:02d}{args.time.tm_sec:02d}/"

    if (not torch.cuda.is_available() and args.device != "cpu"):
        print("No CUDA, using CPU.")
        args.device = "cpu"
    else:
        print("Using CUDA.")

    os.makedirs("result/", exist_ok=True)
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

    args.log.close()

    print("\nTotal time = %f(s)" % (time.time() - start_time))
