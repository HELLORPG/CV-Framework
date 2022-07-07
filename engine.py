# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Train and Evaluation functions, mainly used in main.py.
import torch

from models.build import build_model


def train(config: dict):
    """
    Train the model, using a config.

    Args:
        config: Mainly config.
    """
    model = build_model(config=config)
    model.to(device=torch.device(config["DEVICE"]))

    print("Here")

