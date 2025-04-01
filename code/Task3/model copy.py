import torch
import random
from tqdm import tqdm
import numpy as np

# DTYPE = torch.float16
# DTYPE = torch.float32
DTYPE = torch.float64


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


class LogisticModel():
    def __init__(self, args):
        self.alpha_1 = args.alpha_1
        self.alpha_2 = args.alpha_2
        self.max_iter = args.max_iter
        self.device = args.device
        self.threshold = args.threshold

        self.init_lr = args.lr
        self.gamma = args.gamma

        self.w = None

    def preprocess(self, _x):
        x = torch.tensor(_x, device=self.device, dtype=DTYPE)
        x = torch.concatenate((x, torch.ones((x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)
        return x

    def fit(self, _x, _y):
        x = self.preprocess(_x)
        y = torch.tensor(_y, device=self.device, dtype=DTYPE)
        self.w = torch.zeros((x.shape[1]), device=self.device, dtype=DTYPE)
        self.lr = self.init_lr
        for i in tqdm(range(self.max_iter)):
            pred_y = sigmoid(x @ self.w).reshape(-1)
            dw = ((pred_y - y) @ x).reshape(self.w.shape) / float(_x.shape[0])
            dw += self.alpha_1 * 2.0 * ((self.w > 0).float() - 0.5)
            dw += self.alpha_2 * 2.0 * self.w
            self.w -= self.lr * dw
            # if (i >= (self.max_iter // 2)):
            #     self.lr *= self.gamma
        print(torch.linalg.norm(dw))
        return self

    def predict(self, _x):
        x = self.preprocess(_x)
        x = sigmoid(x @ self.w).reshape(-1)
        return (x >= self.threshold).int().cpu()


class SplitModel():
    def __init__(self, args):
        self.split_data = args.split_data
        self.split_feat = args.split_feat
        self.n_estimator = args.n_estimator
        self.alpha_1 = args.alpha_1
        self.alpha_2 = args.alpha_2
        self.alpha_3 = args.alpha_3
        self.max_iter = args.max_iter
        self.device = args.device
        self.threshold = args.threshold

        self.init_lr = args.lr
        self.gamma = args.gamma

        self.w = None
        self.w_a = None
        self.feat_index = None
        self.num_feat = None

    def fit(self, _x, _y):
        self.feat_index = []
        all_x = torch.tensor(_x, device=self.device, dtype=DTYPE)
        all_x = torch.concatenate((all_x, torch.ones((all_x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)
        all_y = torch.tensor(_y, device=self.device, dtype=DTYPE)
        n = round(self.split_data * _x.shape[0])
        self.num_feat = round(self.split_feat * _x.shape[1])
        x = torch.zeros((self.n_estimator, n, self.num_feat + 1), device=self.device, dtype=DTYPE)
        y = torch.zeros((self.n_estimator, n), device=self.device, dtype=DTYPE)
        for i in range(self.n_estimator):
            index = random.sample(range(_x.shape[0]), n)
            self.feat_index.append(np.unique(sorted(random.sample(range(_x.shape[1]), self.num_feat) + [_x.shape[1]])))
            x[i] = all_x[index][:, self.feat_index[-1]]
            y[i] = all_y[index]
        self.w = torch.zeros((self.n_estimator, self.num_feat + 1, 1), device=self.device, dtype=DTYPE)
        self.lr = self.init_lr
        for i in tqdm(range(self.max_iter)):
            pred_y = sigmoid(x @ self.w).reshape(self.n_estimator, -1)
            dw = ((pred_y - y).reshape(self.n_estimator, 1, -1) @ x).reshape(self.w.shape) / float(n)
            dw += self.alpha_1 * 2.0 * ((self.w > 0).float() - 0.5)
            dw += self.alpha_2 * 2.0 * self.w

            W = self.w[:, :-1].reshape(self.n_estimator, -1)
            S = W @ W.transpose(0, 1)
            do = 2.0 * self.alpha_3 * ((S - torch.diag_embed(torch.diag(S))) @ W)
            # do = 2.0 * self.alpha_3 * ((S - torch.eye(S.shape[0], device=self.device, dtype=DTYPE)) @ W)
            dw[:, :-1, 0] += do
            self.w -= self.lr * dw
            if (i >= (self.max_iter // 2)):
                self.lr *= self.gamma
        print(torch.linalg.norm(dw))
        # W = self.w[:, :-1].reshape(self.n_estimator, -1)
        # S = W @ W.transpose(0, 1)
        # print(S)

        x = torch.zeros((self.n_estimator, _x.shape[0], self.num_feat + 1), device=self.device, dtype=DTYPE)
        y = torch.zeros((self.n_estimator, _x.shape[0]), device=self.device, dtype=DTYPE)
        for i in range(self.n_estimator):
            self.feat_index.append(np.unique(sorted(random.sample(range(_x.shape[1]), self.num_feat) + [_x.shape[1]])))
            x[i] = all_x[:, self.feat_index[-1]]
            y[i] = all_y
        x = sigmoid(x @ self.w).reshape(self.n_estimator, -1).transpose(0, 1)
        x = (x >= self.threshold).int()

        # x = (torch.count_nonzero(x, axis=1) / float(self.n_estimator)).reshape(-1, 1)
        x = torch.concatenate((all_x, x, torch.ones((x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)

        self.w_a = torch.zeros((x.shape[1], 1), device=self.device, dtype=DTYPE)
        self.lr = self.init_lr
        for i in tqdm(range(self.max_iter)):
            pred_y = sigmoid(x @ self.w_a).reshape(-1)
            dw_a = ((pred_y - all_y) @ x).reshape(self.w_a.shape) / float(_x.shape[0])
            dw_a += self.alpha_1 * 2.0 * ((self.w_a > 0).float() - 0.5)
            dw_a += self.alpha_2 * 2.0 * self.w_a
            self.w_a -= self.lr * dw_a
            if (i >= (self.max_iter // 2)):
                self.lr *= self.gamma
        print(torch.linalg.norm(dw_a))
        return self

    def predict(self, _x):
        all_x = torch.tensor(_x, device=self.device, dtype=DTYPE)
        all_x = torch.concatenate((all_x, torch.ones((all_x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)
        x = torch.zeros((self.n_estimator, _x.shape[0], self.num_feat + 1), device=self.device, dtype=DTYPE)
        for i in range(self.n_estimator):
            self.feat_index.append(np.unique(sorted(random.sample(range(_x.shape[1]), self.num_feat) + [_x.shape[1]])))
            x[i] = all_x[:, self.feat_index[-1]]
        x = sigmoid(x @ self.w).reshape(self.n_estimator, _x.shape[0]).transpose(0, 1)
        # x = (x >= self.threshold).int()
        # x = (torch.count_nonzero(x, axis=1) / float(self.n_estimator)).reshape(-1, 1)
        x = torch.concatenate((all_x, x, torch.ones((x.shape[0], 1), device=self.device, dtype=DTYPE)), axis=1)
        x = sigmoid(x @ self.w_a).reshape(-1)
        return (x >= self.threshold).int().cpu()
