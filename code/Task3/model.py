import math
import random
import torch
from tqdm import tqdm
from joblib import Parallel, delayed

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


class DecisionTree():
    def __init__(self, max_depth, min_samples_split, criterion, device):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.device = device
        self.tree = None

    def fit(self, x, y):
        x = torch.tensor(x, device=self.device, dtype=DTYPE)
        y = torch.tensor(y, device=self.device, dtype=torch.int)
        self.tree = self._build_tree(x, y, depth=0)
        return self

    def _build_tree(self, x, y, depth):
        n_samples = x.shape[0]
        n_classes = len(torch.unique(y))

        if (depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1):
            return self._make_leaf_node(y)

        best_feature, best_threshold = self._find_best_split(x, y)
        if best_feature is None:
            return self._make_leaf_node(y)

        mask = x[:, best_feature] <= best_threshold

        left_subtree = self._build_tree(x[mask], y[mask], depth + 1)
        right_subtree = self._build_tree(x[~mask], y[~mask], depth + 1)

        return {"feature": best_feature, "threshold": float(best_threshold),
                "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, x, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in range(x.shape[1]):
            thresholds = torch.unique(x[:, feature])
            gain = self._calculate_information_gain(x, y, feature)
            i = torch.argmax(gain)
            if (gain[i] > best_gain):
                best_gain = gain[i]
                best_feature = feature
                best_threshold = thresholds[i]
            # for i in range(thresholds.shape[0]):
            #     if gain[i] > best_gain:
            #         best_gain = gain[i]
            #         best_feature = feature
            #         best_threshold = thresholds[i]
        return best_feature, best_threshold

    def _calculate_information_gain(self, x, y, feature):
        thresholds = torch.unique(x[:, feature]).reshape(-1, 1)
        parent_impurity = self._calculate_impurity(y.reshape(1, -1))
        mask = (x[:, feature].reshape(1, x.shape[0]).repeat((thresholds.shape[0], 1)) <= thresholds)
        n_left = torch.sum(mask, dim=1)
        n_right = x.shape[0] - n_left
        gain = torch.ones(thresholds.shape[0], device=self.device, dtype=DTYPE) * parent_impurity
        gain -= (n_left / len(y)) * self._calculate_impurity(y.reshape(1, -1) - 2 * y.max() * ~mask)
        gain -= (n_right / len(y)) * self._calculate_impurity(y.reshape(1, -1) - 2 * y.max() * mask)
        gain[n_left == 0] = 0.0
        gain[n_right == 0] = 0.0
        return gain

    def _calculate_impurity(self, y):
        if (self.criterion == "entropy"):
            r = torch.zeros(y.shape[0], device=self.device, dtype=DTYPE)
            s = torch.sum(y >= 0, dim=1)
            for k in torch.unique(y):
                if (k >= 0):
                    probs = torch.sum(y == k, dim=1) / s
                    r -= probs * torch.log2(probs + 1e-10)
            return r
        elif (self.criterion == "gini"):
            r = torch.ones(y.shape[0], device=self.device, dtype=DTYPE)
            s = torch.sum(y >= 0, dim=1)
            for k in torch.unique(y):
                if (k >= 0):
                    probs = torch.sum(y == k, dim=1) / s
                    r -= probs ** 2
            return r

    def _make_leaf_node(self, y):
        counts = torch.bincount(y)
        return {"class": torch.argmax(counts).item()}

    def predict(self, x):
        x = torch.tensor(x, device=self.device, dtype=torch.float32)
        return torch.tensor([self._traverse_tree(t, self.tree) for t in x]).cpu()

    def _traverse_tree(self, x, node):
        if "class" in node:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])


class RandomForest():
    def __init__(self, args):
        self.n_estimator = args.n_estimator
        self.max_depth = args.max_depth
        self.min_samples_split = args.min_samples_split
        self.criterion = args.criterion
        self.device = args.device
        self.threshold = args.threshold
        self.estimator = []
        self.feat_index = []

    def fit(self, x, y):
        for i in tqdm(range(self.n_estimator)):
            index_data = random.sample(range(x.shape[0]), round(0.05 * x.shape[0]))
            index_feat = random.sample(range(x.shape[1]), round(math.sqrt(x.shape[1])))
            self.feat_index.append(index_feat)
            self.estimator.append(DecisionTree(self.max_depth, self.min_samples_split, self.criterion, self.device))
            self.estimator[i].fit(x[index_data][:, index_feat], y[index_data])
        return self

    def predict(self, x):
        res = torch.zeros(self.n_estimator, x.shape[0], device=self.device, dtype=DTYPE)
        for i in tqdm(range(self.n_estimator)):
            res[i] = self.estimator[i].predict(x[:, self.feat_index[i]])
        return (res.sum(dim=0) >= self.threshold).int().cpu()
