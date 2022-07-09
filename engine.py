# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Train and Evaluation functions, mainly used in main.py.
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from models.utils import build_model, save_checkpoint, load_checkpoint
from data.build import build_dataloader
from utils import labels_to_one_hot
from torch.optim import Adam
from logger import MetricLog, Logger
from tqdm import tqdm


def train(config: dict, logger: Logger):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
        logger: A logger.
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
        train_log = train_one_epoch(config=config, model=model,
                                    dataloader=train_dataloader, loss_function=loss_function,
                                    optimizer=optimizer,
                                    epoch=epoch)
        test_log = evaluate_one_epoch(config=config, model=model,
                                      dataloader=test_dataloader, loss_function=loss_function)
        log = MetricLog.concat(metrics=[train_log, test_log])
        logger.show(log, "")
        save_checkpoint(config, model=model,
                        path=os.path.join(config["OUTPUTS"]["OUTPUTS_DIR"], "checkpoint_%d.pth" % epoch),
                        optimizer=optimizer,
                        )

    # print("Here")


def train_one_epoch(config: dict, model: nn.Module,
                    dataloader: DataLoader, loss_function: nn.Module, optimizer: torch.optim,
                    epoch: int):
    """
    Args:
        config: Main config.
        model: Model.
        dataloader: Training dataloader.
        loss_function: Loss function.
        optimizer: Training optimizer.
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


def evaluate(config: dict, logger: Logger):
    """
    Evaluate a model.

    Args:
        config:
        logger:

    Returns:

    """
    model = build_model(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    load_checkpoint(config, model, path=config["EVAL"]["EVAL_MODEL"])

    dataloader = build_dataloader(
        dataset=config["DATA"]["DATASET"],
        root=config["DATA"]["DATA_PATH"],
        split="test",
        bs=config["TRAIN"]["BATCH_SIZE"] * 2
    )

    loss_function = nn.CrossEntropyLoss()

    log = evaluate_one_epoch(config=config, model=model, dataloader=dataloader, loss_function=loss_function)
    logger.show(log)

    return


@torch.no_grad()
def evaluate_one_epoch(config: dict, model: nn.Module, dataloader: DataLoader, loss_function: nn.Module,):
    """

    Args:
        config:
        model:
        dataloader:
        loss_function:

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
