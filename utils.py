# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Some utils.
import yaml
import numpy as np


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
    config = yaml_to_dict("./configs/resnet18_mnist.yaml")


