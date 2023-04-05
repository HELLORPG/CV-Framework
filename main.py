# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Main Function.
import os
import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed
from log.logger import Logger, parser_to_dict
from configs.utils import update_config
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

    # About system.
    parser.add_argument("--device", type=str, help="Device.")

    # About data.
    parser.add_argument("--data-path", type=str, help="Data path.")

    # About evaluation.
    parser.add_argument("--eval-model", type=str, help="Eval model path.")

    # About outputs.
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir.")
    parser.add_argument("--exp-name", type=str, help="Exp name.")

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
    # 有部分在科研工作中引入的经验用于修改代码结构：
    # TODO 1：MetricLog 的 Smooth 长度不应该是无限制的，应该引入一个 max_length 用于限制，将列表改变为一个队列实现。
    # TODO 2：MetricLog 不应该在每一个 Epoch 内进行初始化，容易带来指标抖动问题，应该作为参数传递进入 Train One Epoch 等函数。
    # TODO 3：Logger 可以考虑不从 main 函数中传入 train，可以考虑在 train/eval 等函数内部初始化。
    # TODO 4：应该在每一个 Epoch 内部增加一个统计当前的 Epoch 的指标结果。
    # TODO 5：还是应该不要引入 'Epoch+1' 这类代码，虽然符合直觉，但是可能不符合代码的逻辑。
    # TODO 6：ProcessLog 可以删除，采用考虑手动计算 rest time 的方式来体现时间进度。
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

    # Then, update configs by runtime options, using the different runtime setting.
    main(config=update_config(config=cfg, option=opt))
