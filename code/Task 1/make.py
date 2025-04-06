raising = False

alpha_1 = 0.0
alpha_2 = 0.0
threshold = 0.5
lr = 0.1
gamma = 1.125
tol = 1e-3

n_estimator = 1
max_depth = 10
min_samples_split = 2
min_samples_leaf = 1
n_threshold = 64
criterion = "gini"

with open("logistic_regression.sh", "w") as fp:
    # for alpha_1 in []:
    # for alpha_2 in []:
    # for threshold in []:
    # for lr in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    # for gamma in []:
    # for tol in []:
    # for raising in []:
    for degree in [1, 2, 3]:
        print(
            f"python main.py --model logistic --alpha_1 {alpha_1} --alpha_2 {alpha_2} --threshold {threshold} --lr {lr} --gamma {gamma} --tol {tol} {'--raising' if raising else ''} --degree {degree} --device cuda",
            file=fp)

with open("decision_tree.sh", "w") as fp:
    print(
        f"python main.py --model randomforest --n_estimator {n_estimator} --max_depth {max_depth} --min_samples_split {min_samples_leaf} --n_threshold {n_threshold} --criterion {criterion} {'--raising' if raising else ''} --device cpu",
        file=fp)