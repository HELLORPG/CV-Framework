# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import os
import random

import yaml
import torch
import torch.distributed
import random
import numpy as np


def set_seed(seed: int):
    seed = seed + distributed_rank()    # 用于避免每张卡的随机结果是一样的
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return


def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True


def distributed_rank():
    if not is_distributed():
        return 0
    else:
        return torch.distributed.get_rank()


def is_main_process():
    return distributed_rank() == 0


def distributed_world_size():
    if is_distributed():
        return torch.distributed.get_world_size()
    else:
        raise RuntimeError("'world size' is not available when distributed mode is not started.")


def yaml_to_dict(path: str):
    """
    Read a yaml file into a dict.

    Args:
        path (str): The path of yaml file.

    Returns:
        A dict.
    """
    with open(path) as f:
        return yaml.load(f.read(), yaml.FullLoader)


def labels_to_one_hot(labels: np.ndarray, class_num: int):
    """
    Args:
        labels: Original labels.
        class_num:

    Returns:
        Labels in one-hot.
    """
    return np.eye(N=class_num)[labels].reshape((len(labels), -1))
    # return np.eye(N=class_num)[labels]


if __name__ == '__main__':
    config = yaml_to_dict("../configs/resnet18_mnist.yaml")


