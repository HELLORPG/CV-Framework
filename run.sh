# Test on Single GPU:
echo "Clean >> Single GPU: Clean target dir."
rm -rf ./outputs/single_gpu/
echo "Begin >> Single GPU: Train from scratch."
if ! python main.py --mode train --outputs-dir ./outputs/single_gpu/train/ --use-distributed False --exp-name single_gpu_train --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo "Begin >> Single GPU: Eval the latest model."
if ! python main.py --mode eval --outputs-dir ./outputs/single_gpu/eval/ --use-distributed False --eval-model ./outputs/single_gpu/train/checkpoint_4.pth --exp-name single_gpu_eval --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo "Begin >> Single GPU: Resume training."
if ! python main.py --mode train --outputs-dir ./outputs/single_gpu/resume/ --use-distributed False --resume-model ./outputs/single_gpu/train/checkpoint_3.pth --exp-name single_gpu_resume --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo -e "\033[32m>>>>  Finish all single gpu test!\033[0m"

# 多卡测试
echo "Clean >> Multi GPU: Clean target dir."
rm -rf ./outputs/multi_gpu/
echo "Begin >> Multi GPU: Train from scratch."
if ! python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/multi_gpu/train/ --use-distributed True --exp-name multi_gpu_train --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo "Begin >> Multi GPU: Eval the latest model."
if ! python -m torch.distributed.run --nproc_per_node=2 main.py --mode eval --outputs-dir ./outputs/multi_gpu/eval/ --use-distributed True --eval-model ./outputs/multi_gpu/train/checkpoint_4.pth --exp-name multi_gpu_eval --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo "Begin >> Multi GPU: Resume training."
if ! python -m torch.distributed.run --nproc_per_node=2 main.py --mode train --outputs-dir ./outputs/multi_gpu/resume/ --use-distributed True --resume-model ./outputs/multi_gpu/train/checkpoint_3.pth --exp-name multi_gpu_resume --exp-group framework; then
  echo -e "\033[31m>>>>  Runtime ERROR!\033[0m"
  exit 0
fi
echo -e "\033[32m>>>>  Finish all multi gpu test!\033[0m"
echo -e "\033[32m>>>>  Congratulations!! All test are finished.\033[0m"