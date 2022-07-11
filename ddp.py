# @Author       : Ruopeng Gao
# @Date         : 2022/7/10
# @Description  : DDP learning file.

import torch.distributed as dist
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
args = parser.parse_args()

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank_arg = args.local_rank               # 命令行形式ARGS形式
local_rank_env = int(os.environ['LOCAL_RANK']) # 在利用env初始ENV环境变量形式
local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])

print(f"{rank=}; {local_rank_arg=}; {local_rank_env=}; {local_world_size=}")

# local_rank_get = dist.get_rank()
# local_rank_env = int(os.environ["LOCAL_RANK"])
# local_world_size = dist.get_world_size()
#
# print(f"{rank=}; {local_rank_get=}; {local_rank_env=}; {local_world_size=}")
