# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Train and Evaluation functions, mainly used in main.py.
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from models.build import build_model
from data.build import build_dataloader


def train(config: dict):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
    """
    model = build_model(config=config)
    model.to(device=torch.device(config["DEVICE"]))

    dataloader = build_dataloader(
        dataset=config["DATA"]["DATASET"],
        root=config["DATA"]["DATA_PATH"],
        split="train",
        bs=config["TRAIN"]["BATCH_SIZE"]
    )

    for epoch in config["TRAIN"]["EPOCHS"]:
        train_one_epoch(model=model, dataloader=dataloader, config=config, epoch=epoch)
        pass

    print("Here")


def train_one_epoch(model: nn.Module, dataloader: DataLoader, config: dict, epoch: int):
    """
    Args:
        model: Model.
        dataloader: Training dataloader.
        config: Main config.
        epoch: Current epoch.

    Returns:
        Logs
    """
    for images, labels in dataloader:
        outputs = model(images)

    pass
