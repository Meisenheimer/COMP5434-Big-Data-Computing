python main.py --model DecisionTree --max_depth 10 --n_threshold 64 --criterion gini --device cpu
python main.py --model DecisionTree --max_depth 10 --n_threshold 128 --criterion gini --device cpu
python main.py --model DecisionTree --max_depth 12 --n_threshold 64 --criterion gini --device cpu
python main.py --model DecisionTree --max_depth 10 --n_threshold 64 --criterion entropy --device cpu
python main.py --model DecisionTree --max_depth 10 --n_threshold 64 --criterion gini --feat_selection True --device cpu
