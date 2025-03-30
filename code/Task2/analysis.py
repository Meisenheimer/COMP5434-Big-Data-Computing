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
    x, y, _ = loadData(args.datafile, True)
    print(x.shape)
    print(y.shape)

    color = []
    for target in y:
        if (target == 1):
            color.append("red")
        elif (target == 2):
            color.append("green")
        elif (target == 3):
            color.append("blue")
        elif (target == 4):
            color.append("yellow")
        else:
            color.append("black")

    # PCA
    print("PCA.")
    pca = PCA(n_components=args.nPCA).fit_transform(x, y)
    for i in range(args.nPCA - 1):
        pyplot.clf()
        pyplot.grid()
        pyplot.scatter(pca[:, i], pca[:, i + 1], color=color, s=[1.0] * pca.shape[0], alpha=0.5)
        pyplot.savefig(os.path.join(args.output, f"PCA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")

    # ICA
    print("ICA.")
    ica = FastICA(n_components=args.nICA).fit_transform(x, y)

    for i in range(args.nICA - 1):
        pyplot.clf()
        pyplot.grid()
        pyplot.scatter(ica[:, i], ica[:, i + 1], color=color, s=[1.0] * ica.shape[0], alpha=0.5)
        pyplot.savefig(os.path.join(args.output, f"ICA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")

    # LDA
    print("LDA.")
    lda = LinearDiscriminantAnalysis(n_components=4).fit_transform(x, y)

    for i in range(3):
        pyplot.clf()
        pyplot.grid()
        pyplot.scatter(pca[:, i], pca[:, i + 1], color=color, s=[1.0] * pca.shape[0], alpha=0.5)
        pyplot.savefig(os.path.join(args.output, f"LDA-{i}-{i + 1}.jpg"), dpi=720, bbox_inches="tight")

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
