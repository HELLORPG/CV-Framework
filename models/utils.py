# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : To build a model.
import torch
import torch.nn as nn
import torch.optim as optim

from .resnet18 import ResNet18


def build_model(config: dict):
    return ResNet18(config=config)


def save_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    save_state = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        'states': states
    }

    torch.save(save_state, path)
    return


def load_checkpoint(model: nn.Module, path: str, states: dict = None,
                    optimizer: optim = None, scheduler: optim.lr_scheduler = None):
    load_state = torch.load(path)

    model.load_state_dict(load_state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    return
