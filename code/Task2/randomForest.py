import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

DTYPE = torch.float32
# DTYPE = torch.float64


class DecisionTree():
    def __init__(self, args):
        self.max_depth = args.max_depth
        self.min_samples_split = args.min_samples_split
        self.criterion = args.criterion
        self.device = args.device
        self.tree = None

    def fit(self, x, y):
        x = torch.tensor(x, device=self.device, dtype=DTYPE)
        y = torch.tensor(y, device=self.device, dtype=torch.int)
        self.tree = self._build_tree(x, y, depth=0)

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
        for feature in tqdm(range(x.shape[1])):
            thresholds = torch.unique(x[:, feature])
            gain = self._calculate_information_gain(x, y, feature)
            for i in range(thresholds.shape[0]):
                if gain[i] > best_gain:
                    best_gain = gain[i]
                    best_feature = feature
                    best_threshold = thresholds[i]
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
        return torch.tensor([self._traverse_tree(x, self.tree) for x in x]).cpu()

    def _traverse_tree(self, x, node):
        if "class" in node:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])
