import torch
import numpy as np
from sklearn.metrics import classification_report

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        X = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(torch.unique(y))

        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return self._make_leaf_node(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return self._make_leaf_node(y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": best_feature,
            "threshold": float(best_threshold),
            "left": left_subtree,
            "right": right_subtree
        }

    def _find_best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._calculate_information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_information_gain(self, X, y, feature, threshold):
        parent_impurity = self._calculate_impurity(y)
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        n_left, n_right = left_mask.sum(), right_mask.sum()
        if n_left == 0 or n_right == 0:
            return 0

        child_impurity = (n_left / len(y)) * self._calculate_impurity(y[left_mask]) + \
                         (n_right / len(y)) * self._calculate_impurity(y[right_mask])
        return parent_impurity - child_impurity

    def _calculate_impurity(self, y):
        counts = torch.bincount(y)
        probs = counts / counts.sum()
        if self.criterion == "gini":
            return 1 - torch.sum(probs ** 2)
        elif self.criterion == "entropy":
            return -torch.sum(probs * torch.log2(probs + 1e-10))
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")

    def _make_leaf_node(self, y):
        counts = torch.bincount(y)
        return {"class": torch.argmax(counts).item()}

    def predict(self, X):
        X = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        return torch.tensor([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if "class" in node:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        else:
            return self._traverse_tree(x, node["right"])