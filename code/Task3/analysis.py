import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm


from matplotlib import pyplot
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score

from preprocess import loadData


def init(args: argparse.ArgumentParser) -> None:
    """
    Set the random seed.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)
    return None


def main(args: argparse.Namespace) -> None:
    """
    The algorithm is shown in the slide and report.
    """
    # load the data from csv file.
    print("Loading data.")
    x, y = loadData(args.datafile, True)
    print(x.shape)
    print(y.shape)

    # Preprocessing.
    print("Preprocessing.")

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

    # # ICA
    # print("ICA.")
    # ica = FastICA(n_components=args.nICA).fit_transform(x, y)

    # for i in range(args.nICA - 1):
    #     pyplot.clf()
    #     pyplot.grid()
    #     pyplot.scatter(ica[:, i], ica[:, i + 1], color=color, s=[1.0] * ica.shape[0], alpha=0.5)
    #     pyplot.savefig(os.path.join(args.output, f"ICA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")

    # LDA
    print("LDA.")
    lda = LinearDiscriminantAnalysis().fit_transform(x, y)

    pyplot.clf()
    pyplot.grid()
    pyplot.hist(lda[y == 0], color="r", label="0", alpha=0.5, density=True, bins=25)
    pyplot.vlines(np.percentile(lda[y == 0], (25, 50, 75)), ymin=0, ymax=0.5, color="r", linestyles="--")
    pyplot.hist(lda[y == 1], color="b", label="1", alpha=0.5, density=True, bins=25)
    pyplot.vlines(np.percentile(lda[y == 1], (25, 50, 75)), ymin=0, ymax=0.5, color="b", linestyles="--")
    pyplot.legend()
    pyplot.savefig(os.path.join(args.output, f"LDA.jpg"), dpi=720, bbox_inches="tight")

    lda[lda > 0] = 1
    lda[lda <= 0] = 0
    print(f1_score(y, lda, average="macro"))

    # QDA
    print("QDA.")
    qda = QuadraticDiscriminantAnalysis().fit(x, y).predict(x)
    qda[qda > 0] = 1
    qda[qda <= 0] = 0
    print(f1_score(y, qda, average="macro"))

    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, required=True)

    parser.add_argument("--nPCA", type=int, default=4)
    parser.add_argument("--nICA", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./Result/Analysis/")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    init(args)
    main(args)

    print("Total time = %f(s)" % (time.time() - start_time))
