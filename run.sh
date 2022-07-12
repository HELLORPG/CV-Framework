# 用于测试整个框架的运行
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 单卡测试
echo "Begin >> Single GPU: Train from scratch."
python main.py --mode train --outputs-dir ./outputs/single_gpu/train/ --use-distributed False
echo "Begin >> Single GPU: Eval the latest model."
python main.py --mode eval --outputs-dir ./outputs/single_gpu/eval/ --use-distributed False --eval-model ./outputs/single_gpu/train/checkpoint_10.pth
echo "Begin >> Single GPU: Resume training."
python main.py --mode train --outputs-dir ./outputs/single_gpu/resume/ --use-distributed False --resume-model ./outputs/single_gpu/train/checkpoint_5.pth

# 多卡测试
echo "Begin >> Multi GPU: Train from scratch."
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --outputs-dir ./outputs/multi_gpu/train/ --use-distributed True
echo "Begin >> Multi GPU: Eval the latest model."
python -m torch.distributed.run --nproc_per_node=8 main.py --mode eval --outputs-dir ./outputs/multi_gpu/eval/ --use-distributed True --eval-model ./outputs/multi_gpu/train/checkpoint_10.pth
echo "Begin >> Multi GPU: Resume training."
python -m torch.distributed.run --nproc_per_node=8 main.py --mode train --outputs-dir ./outputs/multi_gpu/resume/ --use-distributed True --resume-model ./outputs/multi_gpu/train/checkpoint_5.pth