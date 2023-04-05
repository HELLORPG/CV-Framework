# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : __init__.py
import torch

from .resnet18 import build as build_resnet18
from utils.utils import distributed_rank


def build_model(config: dict):
    model = build_resnet18(config=config)
    # Choose device:
    if config["AVAILABLE_GPUS"] is not None and config["DEVICE"] == "cuda":
        model.to(device=torch.device(config["DEVICE"], distributed_rank()))
    else:
        model.to(device=torch.device(config["DEVICE"]))
    return model
