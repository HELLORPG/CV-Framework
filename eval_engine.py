# @Author       : Ruopeng Gao
# @Date         : 2023/4/4
# @Description  : Evaluation Processes.
import torch
import torch.distributed
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from models import build_model
from models.utils import load_checkpoint
from data import build_dataset, build_sampler, build_dataloader
from log.logger import Logger, ProgressLogger
from log.log import Metrics
from utils.utils import is_distributed, distributed_rank, labels_to_one_hot


def evaluate(config: dict, logger: Logger):
    """
    Evaluate a model.

    Args:
        config:
        logger:

    Returns:

    """
    model = build_model(config=config)
    # model.to(device=torch.device(config["DEVICE"]))
    load_checkpoint(model, path=config["EVAL_MODEL"])

    test_dataset = build_dataset(
        config=config,
        split="test"
    )
    test_sampler = build_sampler(
        dataset=test_dataset,
        shuffle=False
    )
    test_dataloader = build_dataloader(
        dataset=test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=config["NUM_WORKERS"]
    )

    if is_distributed():
        model = DDP(model, device_ids=[distributed_rank()])

    eval_metrics = evaluate_one_epoch(config=config, model=model, dataloader=test_dataloader, logger=logger)

    if is_distributed():
        torch.distributed.barrier()
    logger.print_metrics(
        metrics=eval_metrics,
        prompt=f"Eval model {config['EVAL_MODEL']}: "
    )
    logger.save_metrics(
        metrics=eval_metrics,
        prompt=f"Eval model {config['EVAL_MODEL']}: ",
        fmt="{global_average:.4f}",
        statistic="global_average",
        global_step=0,
        filename="eval_log.txt",
        file_mode="a"
    )

    return


@torch.no_grad()
def evaluate_one_epoch(config: dict, model: nn.Module, logger: Logger, dataloader: DataLoader):
    model.eval()
    metrics = Metrics()
    if is_distributed():
        device = torch.device(config["DEVICE"], distributed_rank())
    else:
        device = torch.device(config["DEVICE"])

    # with tqdm(total=len(dataloader)) as t:
    process_log = ProgressLogger(total_len=len(dataloader), prompt="Eval")

    for i, batch in enumerate(dataloader):
        images, labels = batch
        outputs = model(images.to(device))
        labels = torch.from_numpy(
            labels_to_one_hot(labels, config["NUM_CLASSES"])).to(device)

        metrics.update(
            "test_acc",
            value=sum(torch.argmax(labels, dim=1).eq(torch.argmax(outputs, dim=1))).item() / len(labels)
        )
        metrics.sync()

        process_log.update(
            step_len=1,
            test_acc=f"{metrics['test_acc'].global_average*100:.2f}"
        )

    return metrics
