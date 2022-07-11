# @Author       : Ruopeng Gao
# @Date         : 2022/7/7
# @Description  : You may have not only one dataset for your model training and evaluation.
#                 I think it is elegant to provide a unified method for different datasets in this file.
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, DistributedSampler
from .mnist import build_mnist_dataloader
from typing import Tuple, Any, Union, Type


def build_dataloader(dataset: str, root: str, split: str, bs: int, num_workers: int = 1) -> \
        tuple[DataLoader[Any], Union[DistributedSampler[Any], RandomSampler, Type[SequentialSampler]]]:
    """
    A unified method to build a dataloader.

    Args:
        dataset: Dataset name, in .yaml config file: DATA.DATASET
        root: Dataset root.
        split: Data split.
        bs: Batch size.
        num_workers:

    Returns:
        A DataLoader.
    """
    if dataset == "MNIST":
        return build_mnist_dataloader(root=root, split=split, bs=bs, num_workers=num_workers)
    else:
        raise RuntimeError("Unknown dataset name '%s'." % dataset)


if __name__ == '__main__':
    dataloader = build_dataloader("MNIST", "../dataset/MNIST", split="train", bs=8)
    for images, labels in dataloader:
        torch.set_printoptions(profile="full")
        print(type(images))
        print(images)
        # print(data[0].shape)
        exit(-1)
