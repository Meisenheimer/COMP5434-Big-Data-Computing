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
        return data[LABELS].to_numpy().astype(float), data["label"].to_numpy().astype(int), data["id"].to_numpy()
    else:
        return data[LABELS].to_numpy().astype(float), data["id"].to_numpy()


def preprocess(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    index = np.zeros((x.shape[0], ))
    for i in range(18):
        index += np.isnan(x[:, i])
        print(i, np.isnan(x[:, i]).sum())
    x = x[index == 0]
    y = y[index == 0]
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
        ('pearson', 'Pearson'),
        ('spearman', 'Spearman'),
        ('kendall', 'Kendall')
    ]

    LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51", "label"]
    df = pd.read_csv(args.datafile).replace("?", "nan").astype(float)[LABELS]
    df = df.dropna(axis=0, how='any')
    for label in range(1, 6):
        tag = f"label-{label}"
        tmp_df = df.copy()
        tmp_df[tag] = tmp_df["label"]
        tmp_df[tag][tmp_df[tag] != label] = 0
        tmp_df[tag][tmp_df[tag] == label] = 1
        tmp_df.drop(columns="label", inplace=True)
        pyplot.figure(figsize=(10, 6))
        for i, (method, title) in enumerate(methods, 1):
            pyplot.subplot(1, 3, i)
            corr = tmp_df.corr(method=method, numeric_only=True)
            sns.heatmap(corr[[tag]].sort_values(by=tag, ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
            pyplot.title(title)
        # pyplot.tight_layout()
        pyplot.savefig(os.path.join(args.output, f"corrcoef-{label}.jpg"), bbox_inches='tight')

    # Hist
    print("Hist")
    LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51"]
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
        for label in range(1, 6):
            h, _, _ = pyplot.hist(data[data_y == label], color=label_color[label], label=str(label), alpha=0.5, density=True, bins=50)
            y_max = max(y_max, max(h))
        for label in range(1, 6):
            pyplot.vlines(np.percentile(data[data_y == label], (25, 50, 75)), ymin=0, ymax=y_max, color=label_color[label], linestyles="--")
        pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"Hist-{LABELS[i]}.jpg"), dpi=720, bbox_inches="tight")
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
