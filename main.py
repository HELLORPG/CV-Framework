# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.
import os
import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed
from log.logger import Logger, parser_to_dict
from configs.utils import update_config, load_super_config
from train_engine import train
from eval_engine import evaluate
# from engine import train, evaluate


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
    parser.add_argument("--super-config-path", type=str)

    # About system.
    parser.add_argument("--device", type=str, help="Device.")

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")

    # About evaluation.
    parser.add_argument("--eval-model", type=str, help="Eval model path.")

    # About outputs.
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir.")
    parser.add_argument("--exp-name", type=str, help="Exp name.")
    parser.add_argument("--exp-group", type=str, help="Exp group, for wandb.")
    parser.add_argument("--use-wandb", type=str, help="Whether use wandb.")

    # About train.
    parser.add_argument("--resume-model", type=str, help="Resume training model path.")

    # Distributed.
    parser.add_argument("--use-distributed", type=str, help="Whether use distributed mode.")

    # Hyperparams.
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)

    return parser.parse_args()


def main(config: dict):
    """
    Main function.

    Args:
        config: Model configs.
    """
    # 在环境变量级别设定可用 GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]   # setting available gpus, like: "0,1,2,3"

    if config["USE_DISTRIBUTED"]:
        # 改变此处顺序会导致 gather 卡死，相关链接：
        # https://i.steer.space/blog/2021/01/pytorch-dist-nccl-backend-allgather-stuck
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())

    logger = Logger(
        logdir=os.path.join(config["OUTPUTS_DIR"], config["MODE"]),
        use_tensorboard=config["USE_TENSORBOARD"],
        use_wandb=config["USE_WANDB"],
        only_main=True,
        config=config
    )
    # Log runtime config.
    if is_main_process():
        logger.print_config(config=config, prompt="Runtime Configs: ")
        logger.save_config(config=config, filename="config.yaml")
        # logger.show(log=config, prompt="Main configs: ")
        # logger.write(config, "config.yaml")

    # set seed
    set_seed(config["SEED"])

    if config["MODE"] == "train":
        train(config=config, logger=logger)
    elif config["MODE"] == "eval":
        evaluate(config=config, logger=logger)
    return


if __name__ == '__main__':
    opt = parse_option()                    # runtime options, a subset of .yaml config file (dict).
    cfg = yaml_to_dict(opt.config_path)     # configs from .yaml file, path is set by runtime options.

    # Loading super config:
    if opt.super_config_path is not None:
        cfg = load_super_config(cfg, opt.super_config_path)
    else:
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    main(config=update_config(config=cfg, option=opt))
