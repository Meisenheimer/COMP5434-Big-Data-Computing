import math
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

DTYPE = torch.float32
# DTYPE = torch.float64


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def generator():
    while True:
        yield


class LogisticModel():
    def __init__(self, args):
        # self.alpha_1 = args.alpha_1
        # self.alpha_2 = args.alpha_2
        # self.max_iter = args.max_iter
        self.device = args.device
        self.threshold = args.threshold

        self.init_lr = args.lr
        self.gamma = args.gamma
        self.tol = args.tol

        self.w = None

    def preprocess(self, _x):
        x = torch.tensor(_x, device=self.device, dtype=DTYPE)
        x = torch.concatenate((x, torch.ones((x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)
        return x

    def fit(self, _x, _y):
        x = self.preprocess(_x)
        y = torch.tensor(_y, device=self.device, dtype=DTYPE)
        self.w = torch.zeros((x.shape[1]), device=self.device, dtype=DTYPE)
        lr = self.init_lr
        bar = tqdm(generator)
        k = 0
        norm = 1.0
        # for i in tqdm(range(self.max_iter)):
        while (norm >= self.tol):
            pred_y = sigmoid(x @ self.w)
            dw = ((pred_y - y) @ x) / float(_x.shape[0])
            # if (self.alpha_1 != 0.0):
            #     dw += self.alpha_1 * 2.0 * ((self.w > 0).float() - 0.5)
            # if (self.alpha_2 != 0.0):
            #     dw += self.alpha_2 * 2.0 * self.w
            norm = float(torch.linalg.norm(dw)) / math.sqrt(dw.shape[0])
            self.w -= (lr / max(1.0, norm + self.tol)) * dw
            if (k % 128 == 0):
                bar.set_postfix({"lr": lr, "norm(Dw)": norm})
            if (k % 2048 == 0):
                lr *= self.gamma
            k += 1
            bar.update(1)
        return self

    def predict(self, _x):
        x = self.preprocess(_x)
        x = sigmoid(x @ self.w).reshape(-1)
        return (x >= self.threshold).int().cpu()
