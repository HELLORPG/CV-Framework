# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.


import yaml
import argparse

from utils import yaml_to_dict


def parse_options():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    The hyperparameters which would influence the results should be included in config file, such as epochs, lr,
    and so on.

    Returns:
        A parser.

    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=False)

    # Config file.
    parser.add_argument("--config-path", type=str, help="Config file path.")

    # About system.
    parser.add_argument("--device", type=str, default="cuda", help="Device.")

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")

    return parser.parse_args()


if __name__ == '__main__':
    options = parse_options()                       # runtime options
    configs = yaml_to_dict(options.config_path)     # configs

