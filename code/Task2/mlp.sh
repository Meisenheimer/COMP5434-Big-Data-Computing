python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 128 --dropout 0.3 --mlp_epoch 200 --device cuda
python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 64 --dropout 0.3 --mlp_epoch 200 --device cuda
python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 256 --dropout 0.3 --mlp_epoch 200 --device cuda
python main.py --model MLP --lr 0.01 --batch_size 1024 --hidden_size 128 --dropout 0.3 --mlp_epoch 200 --device cuda
python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 128 --dropout 0.3 --mlp_epoch 400 --device cuda
python main.py --model MLP --lr 0.01 --batch_size 512 --hidden_size 128 --dropout 0.3 --mlp_epoch 200 --feat_selection True --device cuda
