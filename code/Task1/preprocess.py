import numpy as np
import pandas as pd

LABELS = ["CGPA", "Internships", "Projects", "Workshops/Certifications", "AptitudeTestScore", "SoftSkillsRating", "ExtracurricularActivities", "PlacementTraining", "SSC_Marks", "HSC_Marks"]


def loadData(filename: str, target: bool = False):
    data = pd.read_csv(filename).replace("No", 0).replace("Yes", 1)
    if (target):
        return preprocess(data[LABELS].to_numpy()), data["label"].to_numpy(), data["StudentID"].to_numpy()
    else:
        return preprocess(data[LABELS].to_numpy()), data["StudentID"].to_numpy()


def preprocess(x):
    data = np.zeros((x.shape[0], 18))

    # CGPA 0 -> 0
    data[:, 0] = x[:, 0]
    # Internships 1 -> 1, 2, 3
    data[x[:, 1] == 1, 1] = 1
    data[x[:, 1] == 2, 2] = 1
    data[x[:, 1] == 3, 3] = 1
    # Projects 2 -> 4, 5, 6, 7
    data[x[:, 2] == 0, 4] = 1
    data[x[:, 2] == 1, 5] = 1
    data[x[:, 2] == 2, 6] = 1
    data[x[:, 2] == 3, 7] = 1
    # Workshops/Certifications 3 -> 8, 9, 10, 11
    data[x[:, 3] == 0, 8] = 1
    data[x[:, 3] == 1, 9] = 1
    data[x[:, 3] == 2, 10] = 1
    data[x[:, 3] == 3, 11] = 1
    # AptitudeTestScore 4 -> 12
    data[:, 12] = x[:, 4]
    # SoftSkillsRating 5 -> 13
    data[:, 13] = x[:, 5]
    # ExtracurricularActivities 6 -> 14
    data[:, 14] = x[:, 6]
    # PlacementTraining 7 -> 15
    data[:, 15] = x[:, 7]
    # SSC_Marks 8 -> 16
    data[:, 16] = x[:, 8]
    # HSC_Marks 9 -> 17
    data[:, 17] = x[:, 9]

    return data


if __name__ == "__main__":
    x, y, _ = loadData("./poly-u-comp-5434-20242-project-part-1/train.csv", True)
    print(x.shape)
    print(y.shape)
    data = preprocess(x)
    print(x[0])
    print(data[0])
    print(x[1])
    print(data[1])
    pass
