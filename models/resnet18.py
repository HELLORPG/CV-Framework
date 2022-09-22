# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Create a ResNet18 Model.


import torchvision
from torchvision.models import ResNet18_Weights

import torch.nn as nn
from utils.utils import is_main_process


class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()

        if config["MODEL"]["PRETRAINED"] is None:
            if is_main_process():   # 仅仅在主进程进行参数 Load，可以减少读写开销
                self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
            else:
                self.model = torchvision.models.resnet18(weights=None)
        else:
            raise RuntimeError("Unsupported pretrained '%s'." % config["MODEL"]["PRETRAINED"])

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, config["DATA"]["CLASS_NUM"])

        return

    def forward(self, x):
        return self.model(x)
