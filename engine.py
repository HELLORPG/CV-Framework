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
from logger import MetricLog, ProgressLog
from tqdm import tqdm


def train(config: dict):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
    """
    model = build_model(config=config)
    model.to(device=torch.device(config["DEVICE"]))

    train_dataloader = build_dataloader(
        dataset=config["DATA"]["DATASET"],
        root=config["DATA"]["DATA_PATH"],
        split="train",
        bs=config["TRAIN"]["BATCH_SIZE"]
    )

    test_dataloader = build_dataloader(
        dataset=config["DATA"]["DATASET"],
        root=config["DATA"]["DATA_PATH"],
        split="test",
        bs=config["TRAIN"]["BATCH_SIZE"]*2
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(params=model.parameters(), lr=config["TRAIN"]["LR"])

    for epoch in range(0, config["TRAIN"]["EPOCHS"]):
        train_log = train_one_epoch(model=model, dataloader=train_dataloader, loss_function=loss_function,
                                    optimizer=optimizer,
                                    config=config,
                                    epoch=epoch)
        test_log = evaluate(model=model, dataloader=test_dataloader, loss_function=loss_function,
                            config=config)
        log = MetricLog.concat(metrics=[train_log, test_log])
        print(log.mean_metrics)

    # print("Here")


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
    metric_log = MetricLog(epoch=epoch)

    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            images, labels = batch
            outputs = model(images.to(torch.device(config["DEVICE"])))
            labels = torch.from_numpy(
                labels_to_one_hot(labels, config["DATA"]["CLASS_NUM"])).to(torch.device(config["DEVICE"]))

            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_log.update("train_loss", loss.item(), count=len(labels))
            metric_log.update("train_acc",
                              sum(torch.argmax(labels, dim=1) == torch.argmax(outputs, dim=1)).item() / len(labels),
                              len(labels))
            metric_log.mean()

            t.set_description("Train")
            t.set_postfix(loss="%.3f" % metric_log.mean_metrics["train_loss"],
                          acc="%.2f%%" % (metric_log.mean_metrics["train_acc"] * 100))
            t.update(1)
            # print("\r%s" % metric_log.mean_metrics)
            # print(i, "/", len(dataloader))

    return metric_log


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module,
             config: dict):
    """

    Args:
        model:
        dataloader:
        loss_function:
        config:

    Returns:

    """
    model.eval()
    metric_log = MetricLog()

    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            images, labels = batch
            outputs = model(images.to(torch.device(config["DEVICE"])))
            labels = torch.from_numpy(
                labels_to_one_hot(labels, config["DATA"]["CLASS_NUM"])).to(torch.device(config["DEVICE"]))

            loss = loss_function(outputs, labels)

            metric_log.update("test_loss", loss.item(), count=len(labels))
            metric_log.update("test_acc",
                              sum(torch.argmax(labels, dim=1) == torch.argmax(outputs, dim=1)).item() / len(labels),
                              len(labels))
            metric_log.mean()

            t.set_description("Test")
            t.set_postfix(loss="%.3f" % metric_log.mean_metrics["test_loss"],
                          acc="%.2f%%" % (metric_log.mean_metrics["test_acc"] * 100))
            t.update(1)

    return metric_log
