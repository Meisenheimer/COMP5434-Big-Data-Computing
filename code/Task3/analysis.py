import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats
import seaborn as sns

from matplotlib import pyplot
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score


def loadData(filename: str, target: bool = False) -> tuple:
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    data = pd.read_csv(filename)
    if (target):
        return data[LABELS].to_numpy(), data["label"].to_numpy(), data["id"].to_numpy()
    else:
        return data[LABELS].to_numpy(), data["id"].to_numpy()


def preprocess(x: np.ndarray) -> np.ndarray:
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x *= 2.0
    x -= 1.0
    return x


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def drawPC(name: str, x: np.ndarray, y: np.ndarray) -> None:
    index = random.sample(list(range(x.shape[0])), 2000)
    data = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    pyplot.clf()
    pyplot.figure(figsize=[10, 2])
    pyplot.vlines(range(data.shape[1]), ymin=0, ymax=1.0, color="g", linestyles="--", linewidth=0.5)
    for i in index:
        pyplot.plot(range(data.shape[1]), data[i, :], color='b' if y[i] == 1 else 'r', alpha=0.25, linewidth=0.5)
    pyplot.savefig(os.path.join(args.output, f"{name}.jpg"), dpi=720, bbox_inches="tight")
    pyplot.close()
    return None


def main(args: argparse.Namespace) -> None:
    # load the data from csv file.
    print("Loading data.")
    x, y, _ = loadData(args.datafile, True)
    x = preprocess(x)
    print(x.shape)
    print(y.shape)

    color = []
    for target in y:
        color.append("red" if target == 0 else "blue")

    # Correlation
    print("Correlation")
    methods = [
        ('pearson', 'Pearson Correlation Coeffiecient'),
        ('spearman', 'Spearman Correlation Coeffiecient'),
        ('kendall', 'Kendall Correlation Coeffiecient')
    ]
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "label"]
    df = pd.read_csv(args.datafile)[LABELS]
    pyplot.figure(figsize=(12, 6))
    for i, (method, title) in enumerate(methods, 1):
        pyplot.subplot(1, 3, i)
        corr = df.corr(method=method, numeric_only=True)
        sns.heatmap(corr[['label']].sort_values(by='label', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
        pyplot.title(title)
    # pyplot.tight_layout()
    pyplot.savefig(os.path.join(args.output, f"corrcoef.jpg"), bbox_inches='tight')

    # Parallel Coordinates
    print("Parallel Coordinates.")
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    index = range(x.shape[0])
    index = random.sample(list(range(x.shape[0])), x.shape[0] // 10)
    pyplot.clf()
    pyplot.figure(figsize=[10, 2])
    pyplot.vlines(range(x.shape[1]), ymin=-1.0, ymax=1.0, color="g", linestyles="--", linewidth=0.5)
    pyplot.xticks(range(x.shape[1]), LABELS, rotation=30)
    pyplot.yticks([])
    pyplot.xlabel("Feature")
    for i in index:
        tick = np.arange(0, x.shape[1], 1)
        pyplot.plot(tick, x[i, :], color='b' if y[i] == 1 else 'r', alpha=0.25, linewidth=0.5)
    pyplot.savefig(os.path.join(args.output, f"PC.jpg"), dpi=720, bbox_inches="tight")
    pyplot.close()

    # Hist
    print("Hist")
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]
    for i in range(x.shape[1]):
        index = np.zeros((x.shape[0], ))
        lower, upper = np.percentile(x[:, i], (5, 95))
        index += (x[:, i] > upper)
        index += (x[:, i] < lower)
        data = x[index == 0, i]
        data_y = y[index == 0]
        y_max = 0
        pyplot.clf()
        pyplot.figure(figsize=[5, 3])
        pyplot.grid()
        pyplot.xlabel(f"{LABELS[i]}")
        h, _, _ = pyplot.hist(data[data_y == 0], color="r", label="0", alpha=0.5, bins=50)
        y_max = max(y_max, max(h))
        h, _, _ = pyplot.hist(data[data_y == 1], color="b", label="1", alpha=0.5, bins=50)
        y_max = max(y_max, max(h))
        pyplot.vlines(np.percentile(data[data_y == 0], (25, 50, 75)), ymin=0, ymax=y_max, color="r", linestyles="--")
        pyplot.vlines(np.percentile(data[data_y == 1], (25, 50, 75)), ymin=0, ymax=y_max, color="b", linestyles="--")
        pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"Hist-{LABELS[i]}.jpg"), dpi=720, bbox_inches="tight")
        pyplot.close()
    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, default="./poly-u-comp-5434-20242-project-task-3/train.csv")

    parser.add_argument("--nPCA", type=int, default=4)
    parser.add_argument("--nICA", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./Analysis/")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    init(args)
    main(args)

    print("Total time = %f(s)" % (time.time() - start_time))
