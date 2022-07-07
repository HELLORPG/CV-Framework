# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Train and Evaluation functions, mainly used in main.py.
import torch
import torch.nn as nn

from models.build import build_model


def train(config: dict):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
    """
    model = build_model(config=config)
    model.to(device=torch.device(config["DEVICE"]))

    for epoch in config["TRAIN"]["EPOCHS"]:
        pass

    print("Here")


def train_one_epoch(model: nn.Module, config: dict, epoch: int):
    """
    Args:
        model: Model.
        config: Main config.
        epoch: Current epoch.

    Returns:
        Logs
    """
    pass

