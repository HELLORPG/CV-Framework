# @Author       : Ruopeng Gao
# @Date         : 2023/4/4
# @Description  : Training Process.
import os
import torch
import time
import torch.nn as nn
import torch.distributed

from torch.utils.data import DataLoader
from models import build_model
from models.utils import save_checkpoint, load_checkpoint
from models.criterion import build as build_criterion
from data import build_dataset, build_sampler, build_dataloader
from utils.utils import labels_to_one_hot, is_distributed, distributed_rank
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from log.logger import Logger, ProgressLogger
from log.log import Metrics, TPS
from eval_engine import evaluate_one_epoch

from torch.nn.parallel import DistributedDataParallel as DDP


def train(config: dict, logger: Logger):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
        logger: A log.
    """

    model = build_model(config=config)

    # Dataset:
    train_dataset = build_dataset(
        config=config,
        split="train"
    )
    # For some datasets, it is not possible to test during training, so there is no need for test_dataset.
    test_dataset = build_dataset(
        config=config,
        split="test"
    )

    # Sampler:
    train_sampler = build_sampler(
        dataset=train_dataset,
        shuffle=True
    )
    test_sampler = build_sampler(
        dataset=test_dataset,
        shuffle=False
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=config["BATCH_SIZE"],
        sampler=train_sampler,
        num_workers=config["NUM_WORKERS"]
    )
    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=1,   # for eval, most works set bs to 1.
        sampler=test_sampler,
        num_workers=config["NUM_WORKERS"]
    )

    # Criterion (Loss Function):
    loss_function = build_criterion(config=config)

    # Optimizer:
    optimizer = Adam(params=model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])

    # Scheduler:
    if config["SCHEDULER_TYPE"] == "MultiStep":
        scheduler = MultiStepLR(optimizer, milestones=config["SCHEDULER_MILESTONES"],
                                gamma=config["SCHEDULER_GAMMA"])
    else:
        raise RuntimeError(f"Do not support scheduler type {config['SCHEDULER_TYPE']}.")

    # Train States:
    train_states = {
        "start_epoch": 0
    }

    # For resume:
    if config["RESUME_MODEL"] is not None:  # need to resume from checkpoint
        load_checkpoint(
            model=model,
            path=config["RESUME_MODEL"],
            optimizer=optimizer if config["RESUME_OPTIMIZER"] else None,
            scheduler=scheduler if config["RESUME_SCHEDULER"] else None,
            states=train_states if config["RESUME_STATES"] else None
        )
        # Different processing on scheduler:
        if config["RESUME_SCHEDULER"]:
            scheduler.step()
        else:
            for i in range(0, train_states["start_epoch"]):
                scheduler.step()

    # Distributed, every gpu will share the same parameters.
    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    for epoch in range(train_states["start_epoch"], config["EPOCHS"]):
        epoch_start_timestamp = TPS.timestamp()
        if is_distributed():
            train_sampler.set_epoch(epoch)

        # Train one epoch:
        train_metrics = train_one_epoch(config=config, model=model, logger=logger,
                                        dataloader=train_dataloader, loss_function=loss_function,
                                        optimizer=optimizer, epoch=epoch)
        time_per_epoch = TPS.format(TPS.timestamp() - epoch_start_timestamp)
        logger.print_metrics(
            metrics=train_metrics,
            prompt=f"[Epoch {epoch} Finish] [Total Time: {time_per_epoch}] ",
            fmt="{global_average:.4f}"
        )

        # Eval current epoch:
        test_metrics = evaluate_one_epoch(config=config, model=model, logger=logger,
                                          dataloader=test_dataloader)
        logger.print_metrics(
            metrics=test_metrics,
            prompt=f"[Epoch {epoch} Eval] "
        )

        # Save checkpoint.
        save_checkpoint(model=model,
                        path=os.path.join(config["OUTPUTS_DIR"], f"checkpoint_{epoch}.pth"),
                        states={"start_epoch": epoch+1},
                        optimizer=optimizer,
                        scheduler=scheduler
                        )

        # Next step.
        scheduler.step()

    return


def train_one_epoch(config: dict, model: nn.Module, logger: Logger,
                    dataloader: DataLoader, loss_function: nn.Module, optimizer: torch.optim,
                    epoch: int):
    model.train()
    metrics = Metrics()   # save metrics
    tps = TPS()             # save time per step

    if is_distributed():
        device = torch.device(config["DEVICE"], distributed_rank())
    else:
        device = torch.device(config["DEVICE"])

    # process_log = ProgressLogger(total_len=len(dataloader), prompt="Train %d Epoch" % (epoch + 1))

    for i, batch in enumerate(dataloader):
        iter_start_timestamp = TPS.timestamp()
        images, labels = batch
        outputs = model(images.to(device))
        labels = torch.from_numpy(labels_to_one_hot(labels, config["NUM_CLASSES"])).to(device)

        loss = loss_function(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics.train_loss.update(loss.item())
        metrics["train_acc"].update(
            sum(torch.argmax(labels, dim=1).eq(torch.argmax(outputs, dim=1))).item() / len(labels)
        )
        metrics.sync()

        iter_end_timestamp = TPS.timestamp()
        tps.update(iter_end_timestamp - iter_start_timestamp)
        eta = tps.eta(total_steps=len(dataloader), current_steps=i)

        if (i % config["OUTPUTS_PER_STEP"] == 0) or (i == len(dataloader) - 1):
            logger.print_metrics(
                metrics=metrics,
                prompt=f"[Epoch: {epoch}] [{i}/{len(dataloader)}] [tps: {tps.average:.2f}s] [eta: {TPS.format(eta)}] "
            )

    return metrics
