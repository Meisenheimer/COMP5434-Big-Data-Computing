import math
import random
import torch
from torch import nn
from tqdm import tqdm
from torch import optim

from sklearn.metrics import f1_score

DTYPE_FLT = torch.float32
DTYPE_INT = torch.int8


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def generator():
    while True:
        yield


class LogisticModel():
    def __init__(self, alpha_1, alpha_2, device, threshold, lr, gamma, tol):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.device = device
        self.threshold = threshold

        self.init_lr = lr
        self.gamma = gamma
        self.tol = tol

        self.w = None

    def preprocess(self, _x):
        x = torch.tensor(_x, device=self.device, dtype=DTYPE_FLT)
        x = torch.concatenate((x, torch.ones((x.shape[0], 1), device=self.device, dtype=DTYPE_FLT)), axis=1)
        return x

    def fit(self, _x, _y):
        x = self.preprocess(_x)
        y = torch.tensor(_y, device=self.device, dtype=DTYPE_FLT)
        self.w = torch.zeros((x.shape[1]), device=self.device, dtype=DTYPE_FLT)
        lr = self.init_lr
        bar = tqdm(generator)
        k = 0
        norm = 1.0
        while (norm >= self.tol):
            pred_y = sigmoid(x @ self.w)
            dw = ((pred_y - y) @ x) / float(_x.shape[0])
            if (self.alpha_1 != 0.0):
                dw += self.alpha_1 * 2.0 * ((self.w > 0).float() - 0.5)
            if (self.alpha_2 != 0.0):
                dw += self.alpha_2 * 2.0 * self.w
            norm = float(torch.abs(dw).max())
            # norm = float(torch.linalg.norm(dw)) / math.sqrt(dw.shape[0])
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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class MultilayerPerceptron():
    def __init__(self, args):
        self.lr = args.lr
        self.device = args.device
        self.threshold = args.threshold
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.mlp_epoch = args.mlp_epoch
        self.model = None

    def fit(self, _x, _y):
        x = torch.tensor(_x, dtype=DTYPE_FLT, device=self.device)
        y = torch.tensor(_y, dtype=DTYPE_FLT, device=self.device)
        loss_fn = torch.nn.BCELoss()
        self.model = MLP(_x.shape[1], 1, self.hidden_size, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in tqdm(range(self.mlp_epoch)):
            self.model.train()
            self.model.requires_grad_()
            self.model.zero_grad()
            for index in range(0, x.shape[0], self.batch_size):
                optimizer.zero_grad()
                output = self.model(x[index: index + self.batch_size])
                loss = loss_fn(output.reshape(-1), y[index: index + self.batch_size])
                loss.backward()
                optimizer.step()

            self.model.eval()
            output = self.model(x)
        print(f"Epoch {epoch}: F1 Score = {f1_score(_y, (output >= self.threshold).int().cpu(), average='macro')}")
        self.model.eval()

    def predict(self, _x):
        x = torch.tensor(_x, dtype=DTYPE_FLT, device=self.device)
        y = torch.zeros(_x.shape[0], dtype=DTYPE_FLT, device=self.device)
        output = self.model(x).reshape(-1)
        return (output >= self.threshold).int().cpu()
