python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --lr 1.00 --gamma 1.125 --tol 0.001 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --lr 0.50 --gamma 1.125 --tol 0.001 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --lr 0.25 --gamma 1.125 --tol 0.001 --degree 3 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --lr 0.10 --gamma 1.125 --tol 0.001 --degree 4 --device cuda

python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --lr 1.00 --gamma 1.125 --tol 0.001 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --lr 0.50 --gamma 1.125 --tol 0.001 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --lr 0.25 --gamma 1.125 --tol 0.001 --degree 3 --device cuda
python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --lr 0.10 --gamma 1.125 --tol 0.001 --degree 4 --device cuda

python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --lr 1.00 --gamma 1.125 --tol 0.001 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --lr 0.50 --gamma 1.125 --tol 0.001 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --lr 0.25 --gamma 1.125 --tol 0.001 --degree 3 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --lr 0.10 --gamma 1.125 --tol 0.001 --degree 4 --device cuda

python main.py --model RandomForest --n_estimator 11 --max_depth 16 --criterion gini --n_threshold 64 --split_data_size 0.5 --device cpu
python main.py --model RandomForest --n_estimator 15 --max_depth 16 --criterion gini --n_threshold 64 --split_data_size 0.5 --device cpu
python main.py --model RandomForest --n_estimator 19 --max_depth 16 --criterion gini --n_threshold 64 --split_data_size 0.5 --device cpu

python main.py --model RandomForest --n_estimator 11 --max_depth 16 --criterion entropy --n_threshold 64 --split_data_size 0.5 --device cpu
python main.py --model RandomForest --n_estimator 15 --max_depth 16 --criterion entropy --n_threshold 64 --split_data_size 0.5 --device cpu
python main.py --model RandomForest --n_estimator 19 --max_depth 16 --criterion entropy --n_threshold 64 --split_data_size 0.5 --device cpu
