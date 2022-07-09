# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Train and Evaluation functions, mainly used in main.py.
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from models.build import build_model
from data.build import build_dataloader
from utils import labels_to_one_hot
from torch.optim import Adam


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

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=config["TRAIN"]["LR"])

    for epoch in range(0, config["TRAIN"]["EPOCHS"]):
        train_one_epoch(model=model, dataloader=dataloader, loss_function=loss_function, optimizer=optimizer,
                        config=config, epoch=epoch)
        pass

    print("Here")


def train_one_epoch(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module, optimizer: torch.optim,
                    config: dict, epoch: int):
    """
    Args:
        model: Model.
        dataloader: Training dataloader.
        loss_function: Loss function.
        optimizer: Training optimizer.
        config: Main config.
        epoch: Current epoch.

    Returns:
        Logs
    """
    model.train()

    for images, labels in dataloader:
        optimizer.zero_grad()

        outputs = model(images)
        labels = torch.from_numpy(labels_to_one_hot(labels, config["DATA"]["CLASS_NUM"]))

        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        print(loss.item())

    print("!")
