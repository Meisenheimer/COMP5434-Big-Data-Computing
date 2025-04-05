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
    LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51"]
    data = pd.read_csv(filename).replace("?", "nan")
    if (target):
        return data[LABELS].to_numpy(), data["label"].to_numpy().astype(int), data["id"].to_numpy()
    else:
        return data[LABELS].to_numpy(), data["id"].to_numpy()


def preprocess(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    index = np.zeros((x.shape[0], ))
    for i in range(18):
        index += np.isnan(x[:, i])
        print(i, np.isnan(x[:, i]).sum())
    x = x[index == 0]
    y = y[index == 0]
    print(x)
    print(x.shape)
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    x *= 2.0
    x -= 1.0
    return x, y


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def main(args: argparse.Namespace) -> None:
    # load the data from csv file.
    print("Loading data.")
    x, y, _ = loadData(args.datafile, True)
    x, y = preprocess(x, y)
    print(x.shape)
    print(y.shape)

    label_color = ["", "red", "blue", "green", "black", "yellow"]
    color = []
    for target in y:
        color.append(label_color[target])

    # Correlation
    print("Correlation")
    methods = [
        ('pearson', 'Pearson Correlation Coeffiecient'),
        ('spearman', 'Spearman Correlation Coeffiecient'),
        ('kendall', 'Kendall Correlation Coeffiecient')
    ]

    LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51", "label"]
    df = pd.read_csv(args.datafile).replace("?", "nan").astype(float)[LABELS]
    df = df.dropna(axis=0, how='any')
    print(df)
    pyplot.figure(figsize=(18, 6))
    for i, (method, title) in enumerate(methods, 1):
        pyplot.subplot(1, 3, i)
        corr = df.corr(method=method, numeric_only=True)
        sns.heatmap(corr[['label']].sort_values(by='label', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        pyplot.title(title)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(args.output, f"corrcoef.jpg"), bbox_inches='tight')

    # Parallel Coordinates
    print("Parallel Coordinates.")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    index = range(x.shape[0])
    index = random.sample(list(range(x.shape[0])), x.shape[0] // 10)
    pyplot.clf()
    pyplot.figure(figsize=[10, 2])
    pyplot.vlines(range(x.shape[1]), ymin=-1.0, ymax=1.0, color="g", linestyles="--", linewidth=0.5)
    pyplot.yticks([])
    pyplot.xlabel("Feature")
    tick = np.arange(0, x.shape[1], 1)
    for i in index:
        for label in range(1, 6):
            pyplot.plot(tick, x[i, :], color=label_color[y[i]], alpha=0.25, linewidth=0.5)
    for i in range(x.shape[1]):
        upper = min(float(mean[i] + 3 * std[i]), 1)
        lower = max(float(mean[i] - 3 * std[i]), -1)
        pyplot.hlines(y=(lower, mean[i], upper), xmin=i - 0.5, xmax=i + 0.5, color="g", linestyles="--", linewidth=0.5)
    pyplot.savefig(os.path.join(args.output, f"PC.jpg"), dpi=720, bbox_inches="tight")
    pyplot.close()

    # Hist
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    index = np.zeros((x.shape[0], ))
    for i in range(x.shape[1]):
        upper = float(mean[i] + 3 * std[i])
        lower = float(mean[i] - 3 * std[i])
        index += (x[:, i] > upper)
        index += (x[:, i] < lower)
    x = x[index == 0]
    y = y[index == 0]
    print(x.shape)
    print(y.shape)
    for i in range(x.shape[1]):
        data = x[:, i]
        pyplot.clf()
        pyplot.figure(figsize=[10, 3])
        pyplot.grid()
        # pyplot.xlim([-1, 1])
        for label in range(1, 6):
            pyplot.hist(data[y == label], color=label_color[label], label=str(label), alpha=0.5, density=True, bins=50)
            pyplot.vlines(np.percentile(data[y == label], (25, 50, 75)), ymin=0, ymax=0.9, color="r", linestyles="--")
        pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"Hist-{i}.jpg"), dpi=720, bbox_inches="tight")
        pyplot.close()
    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, default="./poly-u-comp-5434-20242-project-task-2/train.csv")

    parser.add_argument("--nPCA", type=int, default=4)
    parser.add_argument("--nICA", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./Analysis/")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    init(args)
    main(args)

    print("Total time = %f(s)" % (time.time() - start_time))
