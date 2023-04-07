# Try different hyperparams:
rm -rf ./outputs/hyperparams/
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp1/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 256 --lr 0.0001
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp2/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 512 --lr 0.0001
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp3/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 128 --lr 0.0001
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp4/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 256 --lr 0.0004
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp5/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 256 --lr 0.0002
python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/hyperparams/exp6/ --use-distributed True --exp-name try_hyperparams --exp-group hyperparams --batch-size 256 --lr 0.00005