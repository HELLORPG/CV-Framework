# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : To build a model.
import torch
import torch.nn as nn
import torch.optim as optim

from .resnet18 import ResNet18


def build_model(config: dict):
    return ResNet18(config=config)


def save_checkpoint(config: dict, model: nn.Module, path: str,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None,):
    save_state = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "lr_scheduler": None if scheduler is None else scheduler.state_dict(),
        'config': config
    }

    torch.save(save_state, path)
    return


def load_checkpoint(config: dict, model: nn.Module, path: str,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    load_state = torch.load(path)
    model.load_state_dict(load_state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    return
