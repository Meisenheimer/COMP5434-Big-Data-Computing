alpha_1 = 0.0
alpha_2 = 0.0
threshold = 0.5
lr = 0.1
gamma = 1.125
tol = 1e-3
raising = False
degree = 1

with open("run.sh", "w") as fp:
    # for alpha_1 in []:
    # for alpha_2 in []:
    # for threshold in []:
    # for lr in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    # for gamma in []:
    # for tol in []:
    # for raising in []:
    for degree in [1, 2, 3, 4]:
        print(f"python main.py --model MineLogistic --alpha_1 {alpha_1} --alpha_2 {alpha_2} --threshold {threshold} --lr {lr} --gamma {gamma} --tol {tol}{'--raising' if raising else ''} --degree {degree}", file=fp)
