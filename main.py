# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.


import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process
from logger import Logger, parser_to_dict
from configs.config import update_config
from engine import train, evaluate


def parse_option():
    """
    Build a parser that can set up runtime options, such as choose device, data path, and so on.
    Every option in this parser should appear in .yaml config file (like ./configs/resnet18_mnist.yaml),
    except --config-path.

    Returns:
        A parser.

    """
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    # Running mode, Training? Evaluation? or ?
    parser.add_argument("--mode", type=str, help="Running mode.")

    # Config file.
    parser.add_argument("--config-path", type=str, help="Config file path.",
                        default="./configs/resnet18_mnist.yaml")

    # About system.
    parser.add_argument("--device", type=str, help="Device.")

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")

    # About evaluation.
    parser.add_argument("--eval-model", type=str, help="Eval model path.")

    # About outputs.
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir.")

    # About train.
    parser.add_argument("--resume-model", type=str, help="Resume training model path.")

    # Distributed.
    parser.add_argument("--use-distributed", type=str, help="Whether use distributed mode.")

    return parser.parse_args()


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    logger = Logger(logdir=config["OUTPUTS"]["OUTPUTS_DIR"])

    if config["DISTRIBUTED"]["USE_DISTRIBUTED"]:
        torch.distributed.init_process_group("nccl")

    # Logging options and configs.
    if is_main_process():
        logger.show(log=config, prompt="Main configs: ")
        logger.write(config, "config.yaml")

    if config["MODE"] == "train":
        train(config=config, logger=logger)
    elif config["MODE"] == "eval":
        evaluate(config=config, logger=logger)
    return


if __name__ == '__main__':
    opt = parse_option()                  # runtime options
    cfg = yaml_to_dict(opt.config_path)   # configs

    # Merge parser option and .yaml config, then run main function.
    main(config=update_config(config=cfg, option=opt))
