# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.


import argparse

from utils import yaml_to_dict
from logger import Logger, parser_to_dict


def parse_options():
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


def main(options, configs):
    """
    Main function.

    Args:
        options: Runtime options.
        configs: Model configs.
    """
    logger = Logger(logdir=options.outputs_dir)

    # Logging options and configs.
    logger.log(log=configs, prompt="Model configs: ")
    logger.log(log=parser_to_dict(options), prompt="Runtime options: ")
    logger.log_dict_to_file(log=parser_to_dict(options), filename="options.json")
    logger.log_dict_to_file(log=configs, filename="configs.json")

    if options.mode == "train":
        pass
    elif options.mode == "eval":
        pass
    return


if __name__ == '__main__':
    opts = parse_options()                  # runtime options
    cfgs = yaml_to_dict(opts.config_path)   # configs

    main(opts, cfgs)

