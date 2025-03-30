import pandas as pd

LABELS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"]


def loadData(filename: str, target: bool = False):
    data = pd.read_csv(filename)
    if (target):
        return data[LABELS].to_numpy(), data["label"].to_numpy(), data["id"].to_numpy()
    else:
        return data[LABELS].to_numpy(), data["id"].to_numpy()


if __name__ == "__main__":
    x, y = loadData("./poly-u-comp-5434-20242-project-task-3/train.csv", True)
    print(x.shape)
    print(y.shape)
    pass
