python main.py --model DecisionTree --max_depth 12 --n_threshold 64 --criterion gini  --device cpu
python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 256 --dropout 0.3 --mlp_epoch 200 --device cuda
