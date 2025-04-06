import math
import random
import torch
from torch import nn
from tqdm import tqdm
from torch import optim

from sklearn.metrics import f1_score

DTYPE_FLT = torch.float32
DTYPE_INT = torch.int8


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


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
            # nn.Sigmoid()
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
        self.num_classes = args.num_classes  # number of different labels
        self.model = None

    def fit(self, _x, _y):
        y = _y - 1
        x = torch.tensor(_x, dtype=DTYPE_FLT, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model = MLP(_x.shape[1], self.num_classes, self.hidden_size, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        bar = tqdm(range(self.mlp_epoch))
        for epoch in bar:
            self.model.train()
            self.model.requires_grad_()
            self.model.zero_grad()
            for index in range(0, x.shape[0], self.batch_size):
                optimizer.zero_grad()
                output = self.model(x[index: index + self.batch_size])
                loss = loss_fn(output, y[index: index + self.batch_size])
                loss.backward()
                optimizer.step()

            self.model.eval()
            #     output = self.model(x)
            #     preds = torch.argmax(output, dim=1) + 1
            # print(f"Epoch {epoch}: F1 Score = {f1_score(_y, (preds >= self.threshold).int().cpu(), average='macro')}")
            with torch.no_grad():
                output = self.model(x)
                preds = torch.argmax(output, dim=1) + 1
                f1 = f1_score(_y, preds.cpu(), average='macro')
                # print(f"Epoch {epoch}: F1 Score = {f1}")
            bar.set_postfix({"F1 Score": float(f1)})
            bar.update(1)
        self.model.eval()

    def predict(self, _x):
        x = torch.tensor(_x, dtype=DTYPE_FLT, device=self.device)
        # y = torch.zeros(_x.shape[0], dtype=DTYPE_FLT, device=self.device)
        # output = self.model(x).reshape(-1)
        # output = self.model(x)
        # preds = torch.argmax(output, dim=1) + 1
        # return (preds >= self.threshold).int().cpu()
        with torch.no_grad():
            output = self.model(x)
            preds = torch.argmax(output, dim=1) + 1  # 转换回1~5
        return preds.cpu().numpy()
