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


def preprocess(x: np.ndarray, args: argparse.ArgumentParser) -> np.ndarray:
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))


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
    x = preprocess(x, args)
    print(x.shape)
    print(y.shape)

    color = []
    for target in y:
        color.append("red" if target == 0 else "blue")

    # # PCA
    # print("PCA.")
    # pca = PCA(n_components=args.nPCA).fit_transform(x, y)
    # for i in range(args.nPCA - 1):
    #     pyplot.clf()
    #     pyplot.grid()
    #     pyplot.scatter(pca[:, i], pca[:, i + 1], color=color, s=[1.0] * pca.shape[0], alpha=0.5)
    #     pyplot.savefig(os.path.join(args.output, f"PCA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")
    #     pyplot.close()
    # drawPC("PCA-PC", pca, y)

    # # ICA
    # print("ICA.")
    # ica = FastICA(n_components=args.nICA).fit_transform(x, y)

    # for i in range(args.nICA - 1):
    #     pyplot.clf()
    #     pyplot.grid()
    #     pyplot.scatter(ica[:, i], ica[:, i + 1], color=color, s=[1.0] * ica.shape[0], alpha=0.5)
    #     pyplot.savefig(os.path.join(args.output, f"ICA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")
    #     pyplot.close()
    # drawPC("ICA-PC", ica, y)

    # # LDA
    # print("LDA.")
    # lda = LinearDiscriminantAnalysis().fit_transform(x, y)

    # lda[lda >= 6] = 6
    # lda[lda <= -6] = -6

    # print(lda[y == 0].min())
    # print(lda[y == 0].max())
    # print(lda[y == 0].min())
    # print(lda[y == 0].max())

    # pyplot.clf()
    # pyplot.figure(figsize=[10, 3])
    # pyplot.grid()
    # pyplot.xlim([-5, 5])
    # pyplot.hist(lda[y == 0], color="r", label="0", alpha=0.5, density=True, bins=60)
    # pyplot.vlines(np.percentile(lda[y == 0], (25, 50, 75)), ymin=0, ymax=0.9, color="r", linestyles="--")
    # pyplot.hist(lda[y == 1], color="b", label="1", alpha=0.5, density=True, bins=60)
    # pyplot.vlines(np.percentile(lda[y == 1], (25, 50, 75)), ymin=0, ymax=0.9, color="b", linestyles="--")
    # pyplot.legend()
    # pyplot.savefig(os.path.join(args.output, f"LDA.jpg"), dpi=720, bbox_inches="tight")
    # pyplot.close()

    # for i in range(x.shape[1]):
    #     data = x[:, i].reshape(-1, 1)
    #     lda = LinearDiscriminantAnalysis()
    #     qda = QuadraticDiscriminantAnalysis()
    #     lda.fit(data, y)
    #     qda.fit(data, y)
    #     print(i, f1_score(y, lda.predict(data), average="macro"), f1_score(y, qda.predict(data), average="macro"))

    # Correlation
    print("Correlation")
    methods = [
        ('pearson', 'Pearson Correlation Coeffiecient'),
        ('spearman', 'Spearman Correlation Coeffiecient'),
        ('kendall', 'Kendall Correlation Coeffiecient')
    ]
    LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "label"]
    df = pd.read_csv(args.datafile)[LABELS]
    pyplot.figure(figsize=(18, 6))
    for i, (method, title) in enumerate(methods, 1):
        pyplot.subplot(1, 3, i)
        corr = df.corr(method=method, numeric_only=True)
        sns.heatmap(corr[['label']].sort_values(by='label', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        pyplot.title(title)
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(args.output, f"corrcoef.jpg"), bbox_inches='tight')

    # print("Correlation")
    # corrcoef = np.corrcoef(x.transpose(), y=y)
    # pyplot.clf()
    # pyplot.grid()
    # pyplot.imshow(corrcoef)
    # pyplot.colorbar()
    # pyplot.savefig(os.path.join(args.output, f"corrcoef.jpg"), dpi=720, bbox_inches="tight")
    # pyplot.close()

    # Parallel Coordinates
    print("Parallel Coordinates.")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    index = range(x.shape[0])
    index = random.sample(list(range(x.shape[0])), x.shape[0] // 10)
    data = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    pyplot.clf()
    pyplot.figure(figsize=[10, 2])
    pyplot.vlines(range(x.shape[1]), ymin=0, ymax=1.0, color="g", linestyles="--", linewidth=0.5)
    for i in index:
        tick = np.arange(0, x.shape[1], 1)
        pyplot.plot(tick, data[i, :], color='b' if y[i] == 1 else 'r', alpha=0.25, linewidth=0.5)
    for i in range(x.shape[1]):
        upper = min(float(mean[i] + 3 * std[i]), 1)
        lower = max(float(mean[i] - 3 * std[i]), 0)
        pyplot.hlines(y=(upper, mean[i], lower), xmin=i - 0.5, xmax=i + 0.5, color="g", linestyles="--", linewidth=0.5)
    pyplot.savefig(os.path.join(args.output, f"PC.jpg"), dpi=720, bbox_inches="tight")
    pyplot.close()

    # # Hist
    # for i in range(x.shape[1]):
    #     data = x[:, i]
    #     data[data >= 4] = 4
    #     data[data <= -4] = -4
    #     print(i, data[y == 0].min(), data[y == 0].max(), data[y == 0].min(), data[y == 0].max())
    #     pyplot.clf()
    #     pyplot.figure(figsize=[10, 3])
    #     pyplot.grid()
    #     pyplot.xlim([-3, 3])
    #     pyplot.hist(data[y == 0], color="r", label="0", alpha=0.5, density=True, bins=60)
    #     pyplot.vlines(np.percentile(data[y == 0], (25, 50, 75)), ymin=0, ymax=0.9, color="r", linestyles="--")
    #     pyplot.hist(data[y == 1], color="b", label="1", alpha=0.5, density=True, bins=60)
    #     pyplot.vlines(np.percentile(data[y == 1], (25, 50, 75)), ymin=0, ymax=0.9, color="b", linestyles="--")
    #     pyplot.legend()
    #     pyplot.savefig(os.path.join(args.output, f"Hist-{i}.jpg"), dpi=720, bbox_inches="tight")
    #     pyplot.close()
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
