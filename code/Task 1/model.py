import math
import random
import torch
from tqdm import tqdm

DTYPE_FLT = torch.float32
DTYPE_INT = torch.int8


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def generator():
    while True:
        yield


class LogisticModel():
    def __init__(self, args):
        self.alpha_1 = args.alpha_1
        self.alpha_2 = args.alpha_2
        self.device = args.device
        self.threshold = args.threshold

        self.init_lr = args.lr
        self.gamma = args.gamma
        self.tol = args.tol

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


class DecisionTree():
    def __init__(self, max_depth, min_samples_split, min_samples_leaf, criterion, device, n_threshold):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.device = device
        self.n_threshold = n_threshold
        self.tree = None

    def fit(self, x, y):
        x = torch.tensor(x, device=self.device, dtype=DTYPE_FLT)
        y = torch.tensor(y, device=self.device, dtype=DTYPE_INT)
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

        if (mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf):
            return self._make_leaf_node(y)

        left_subtree = self._build_tree(x[mask], y[mask], depth + 1)
        right_subtree = self._build_tree(x[~mask], y[~mask], depth + 1)

        return {"feature": best_feature, "threshold": float(best_threshold),
                "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, x, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        for feature in range(x.shape[1]):
            thresholds = torch.unique(x[:, feature])
            if (thresholds.shape[0] >= (self.n_threshold // 2)):
                thresholds = torch.linspace(x[:, feature].min(), x[:, feature].max(), self.n_threshold, device=self.device, dtype=DTYPE_FLT)
            else:
                tmp, _ = torch.sort(thresholds)
                if (tmp.shape[0] > 1):
                    thresholds = (tmp[1:] + tmp[:-1]) / 2.0
            gain = self._calculate_information_gain(x, y.reshape(1, -1), feature, thresholds)
            tmp = torch.where(gain == gain.max())[0]
            i = tmp[random.randint(0, tmp.shape[0] - 1)]
            if (gain[i] > best_gain):
                best_gain = gain[i]
                best_feature = feature
                best_threshold = thresholds[i]
        return best_feature, best_threshold

    def _calculate_information_gain(self, x, y, feature, thresholds):
        thresholds = thresholds.reshape(-1, 1)
        parent_impurity = self._calculate_impurity(y, torch.ones_like(y, dtype=torch.bool, device=self.device))
        mask = (x[:, feature].reshape(1, x.shape[0]).repeat((thresholds.shape[0], 1)) <= thresholds)
        n_left = torch.sum(mask, dim=1)
        n_right = x.shape[0] - n_left
        gain = torch.ones(thresholds.shape[0], device=self.device, dtype=DTYPE_FLT) * parent_impurity
        tmp_y = y.repeat((thresholds.shape[0], 1))
        gain -= (n_left / y.shape[1]) * self._calculate_impurity(tmp_y, mask)
        gain -= (n_right / y.shape[1]) * self._calculate_impurity(tmp_y, ~mask)
        gain[n_left == 0] = 0.0
        gain[n_right == 0] = 0.0
        return gain

    def _calculate_impurity(self, y, mask):
        tmp_y = y.clone()
        tmp_y[~mask] = -1
        if (self.criterion == "entropy"):
            r = torch.zeros(y.shape[0], device=self.device, dtype=DTYPE_FLT)
            s = torch.sum(mask, dim=1)
            for k in torch.unique(y):
                probs = torch.sum((tmp_y == k), dim=1) / s
                r -= probs * torch.log2(probs + 1e-10)
            return r
        elif (self.criterion == "gini"):
            r = torch.ones(y.shape[0], device=self.device, dtype=DTYPE_FLT)
            s = torch.sum(mask, dim=1)
            for k in torch.unique(y):
                probs = torch.sum((tmp_y == k), dim=1) / s
                r -= probs ** 2
            return r

    def _make_leaf_node(self, y):
        counts = torch.bincount(y)
        return {"class": torch.argmax(counts).item()}

    def predict(self, x):
        x = torch.tensor(x, device=self.device, dtype=DTYPE_FLT)
        return torch.tensor([self._traverse_tree(t, self.tree) for t in x]).cpu()

    def _traverse_tree(self, x, node):
        if "class" in node:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
