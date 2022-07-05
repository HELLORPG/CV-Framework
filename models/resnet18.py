# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Create a ResNet18 Model.


import torchvision

import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()

        if config["MODEL"]["PRETRAINED"] == "None":
            self.model = torchvision.models.resnet18(weights=None)
        else:
            raise RuntimeError("Unsupported pretrained '%s'." % config["MODEL"]["PRETRAINED"])

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, config["DATA"]["CLASS_NUM"])

        return

    def forward(self, x):
        return self.model(x)
