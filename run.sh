# 用于测试整个框架的运行
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.run --nproc_per_node=7 main.py --mode train --outputs-dir ./outputs/dis_tmp --use-distributed True
