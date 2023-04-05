# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : To build a model.
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim

from .resnet18 import ResNet18
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import is_distributed, distributed_rank, is_main_process




def get_model(model):
    return model.module if is_distributed() else model


def save_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    if is_main_process():
        model = get_model(model)
        save_state = {
            "model": model.state_dict(),
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            'states': states
        }
        torch.save(save_state, path)
    else:
        pass
    return


def load_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    load_state = torch.load(path)
    if is_main_process():
        model.load_state_dict(load_state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    return
