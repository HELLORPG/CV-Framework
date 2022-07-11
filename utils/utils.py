# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import yaml
import torch.distributed
import numpy as np


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
    return np.eye(N=class_num)[labels]


if __name__ == '__main__':
    config = yaml_to_dict("../configs/resnet18_mnist.yaml")


