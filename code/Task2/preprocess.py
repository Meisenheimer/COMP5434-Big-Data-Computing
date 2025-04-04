import numpy as np
import pandas as pd

LABELS = ["X01", "Y01", "Z01", "X11", "Y11", "Z11", "X21", "Y21", "Z21", "X31", "Y31", "Z31", "X41", "Y41", "Z41", "X51", "Y51", "Z51"]


def loadData(filename: str, target: bool = False) -> tuple:
    data = pd.read_csv(filename).replace("?", "nan").astype(float)
    if (target):
        return preprocess(data[LABELS].to_numpy()), data["label"].to_numpy(), data["id"].to_numpy()
    else:
        return preprocess(data[LABELS].to_numpy()), data["id"].to_numpy()


def preprocess(x: np.ndarray) -> np.ndarray:
    data = np.nanmean(x, axis=0)
    for i in range(18):
        # x[np.isnan(x[:, i]), i] = 0.0
        x[np.isnan(x[:, i]), i] = data[i]
    return x


if __name__ == "__main__":
    x, y, _ = loadData("./poly-u-comp-5434-20242-project-task-2/train.csv", True)
    print(x.shape)
    print(y.shape)
    print(x[1])
    print(x[2])
    x = preprocess(x)
    print(x[1])
    print(x[2])
    pass
