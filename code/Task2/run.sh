python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 3 --feat_selection True --device cuda
python main.py --model DecisionTree --max_depth 16 --criterion gini --n_threshold 32 --device cpu --feat_selection True
