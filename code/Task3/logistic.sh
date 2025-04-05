python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 3 --device cuda

python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0001 --alpha_2 0.0 --degree 3 --device cuda

python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --degree 1 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --degree 2 --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0001 --degree 3 --device cuda

python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 1 --feat_selection True --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 2 --feat_selection True --device cuda
python main.py --model Logistic --alpha_1 0.0 --alpha_2 0.0 --degree 3 --feat_selection True --device cuda
