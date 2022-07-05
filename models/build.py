# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : To build a model.

from .resnet18 import ResNet18


def build_model(config: dict):
    return ResNet18(config=config)

