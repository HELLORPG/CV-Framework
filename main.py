# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.


import argparse

from utils import yaml_to_dict
from logger import Logger, parser_to_dict


def parse_option():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    The hyperparameters which would influence the results should be included in config file, such as epochs, lr,
    and so on.

    Returns:
        A parser.

    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=False)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode.")

    # Config file.
    parser.add_argument("--config-path", type=str, help="Config file path.",
                        default="./configs/resnet18_mnist.yaml")

    # About system.
    parser.add_argument("--device", type=str, help="Device.",
                        default="cuda")

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")

    # About evaluation.
    parser.add_argument("--eval-model", type=str, help="Eval model path.")

    # About outputs.
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir.",
                        default="./outputs/")

    return parser.parse_args()


def main(config, option):
    """
    Main function.

    Args:
        config: Model configs.
        option: Runtime options.
    """
    logger = Logger(logdir=option.outputs_dir)

    # Logging options and configs.
    logger.log(log=config, prompt="Model configs: ")
    logger.log(log=parser_to_dict(option), prompt="Runtime options: ")
    logger.log_dict_to_file(log=parser_to_dict(option), filename="options.json")
    logger.log_dict_to_file(log=config, filename="configs.json")

    if option.mode == "train":
        pass
    elif option.mode == "eval":
        pass
    return


if __name__ == '__main__':
    opt = parse_option()                  # runtime options
    cfg = yaml_to_dict(opt.config_path)   # configs

    main(cfg, opt)

