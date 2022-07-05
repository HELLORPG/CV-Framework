# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Create a ResNet18 Model.


import torchvision

import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, configs):
        super(ResNet18, self).__init__()

        self.model = torchvision.models.resnet18(pretrained=False)

    def forward(self, x):
        return self.model(x)
