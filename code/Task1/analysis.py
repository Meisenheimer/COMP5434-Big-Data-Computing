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


def loadData(filename: str, target: bool = False) -> tuple:
    LABELS = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
    data = pd.read_csv(filename).replace("Yes", 1).replace("No", 0)
    if (target):
        return data[LABELS].to_numpy(), data["label"].to_numpy(), data["StudentID"].to_numpy()
    else:
        return data[LABELS].to_numpy(), data["StudentID"].to_numpy()


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
    LABELS = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks", "label"]
    df = pd.read_csv(args.datafile)[LABELS]
    pyplot.figure(figsize=(18, 6))
    for i, (method, title) in enumerate(methods, 1):
        pyplot.subplot(1, 3, i)
        corr = df.corr(method=method, numeric_only=True)
        sns.heatmap(corr[['label']].sort_values(by='label', ascending=False), annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=False)
        pyplot.title(title)
    # pyplot.tight_layout()
    pyplot.savefig(os.path.join(args.output, f"corrcoef.jpg"), bbox_inches='tight')

    # Parallel Coordinates
    print("Parallel Coordinates.")
    LABELS = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    index = range(x.shape[0])
    index = random.sample(list(range(x.shape[0])), x.shape[0])
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
    LABELS = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
    LABELS2 = ["CGPA", "Internships", "Projects", "WorkshopsCertifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]
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
        h, _, _ = pyplot.hist(data[data_y == 0], color="r", label="0", alpha=0.5, bins=20)
        y_max = max(y_max, max(h))
        h, _, _ = pyplot.hist(data[data_y == 1], color="b", label="1", alpha=0.5, bins=20)
        y_max = max(y_max, max(h))
        pyplot.vlines(np.percentile(data[data_y == 0], (25, 50, 75)), ymin=0, ymax=y_max, color="r", linestyles="--")
        pyplot.vlines(np.percentile(data[data_y == 1], (25, 50, 75)), ymin=0, ymax=y_max, color="b", linestyles="--")
        pyplot.legend()
        pyplot.savefig(os.path.join(args.output, f"Hist-{LABELS2[i]}.jpg"), dpi=720, bbox_inches="tight")
        pyplot.close()
    return None


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, default="./poly-u-comp-5434-20242-project-task-1/train.csv")

    parser.add_argument("--nPCA", type=int, default=4)
    parser.add_argument("--nICA", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="./Analysis/")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    init(args)
    main(args)

    print("Total time = %f(s)" % (time.time() - start_time))
