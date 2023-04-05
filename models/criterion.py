# @Author       : Ruopeng Gao
# @Date         : 2023/4/4
# @Description  : Loss function. For some tasks, the Loss Function is more complicated.
import torch.nn as nn


def build(config: dict):
    return nn.CrossEntropyLoss()
