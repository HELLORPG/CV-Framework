# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml

from tqdm import tqdm
from typing import List, Any
from torch.utils import tensorboard as tb

from log.log import Metrics
from utils.utils import is_main_process


class ProgressLogger:
    def __init__(self, total_len: int, prompt: str = None, only_main: bool = True):
        """
        初始化一个进度日志。

        Args:
            total_len:
            prompt:
            only_main: 只对主进程生效。
        """
        self.only_main = only_main
        self.is_activate = (self.only_main and is_main_process()) or (self.only_main is False)

        if self.is_activate:
            self.total_len = total_len
            self.tqdm = tqdm(total=total_len)
            self.prompt = prompt
        else:
            self.total_len = None
            self.tqdm = None
            self.prompt = None

    def update(self, step_len: int, **kwargs: Any):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tqdm.set_description(self.prompt)
            self.tqdm.set_postfix(**kwargs)
            self.tqdm.update(step_len)
        else:
            return


class Logger:
    """
    Log information.
    """
    def __init__(self, logdir: str, use_tensorboard: bool = True, use_wandb: bool = True, only_main: bool = True):
        """
        Create a log.

        Args:
            logdir (str): Logger outputs path.
            use_tensorboard: Whether output tensorboard files.
            use_wandb: Whether output WandB files.
            only_main: Only in the main process.
        """
        self.only_main = only_main
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.is_activate = (self.only_main and is_main_process()) or (self.only_main is False)

        self.logdir = None
        self.tb_writer = None
        if self.is_activate:
            # init the logdir.
            self.logdir = logdir
            os.makedirs(self.logdir, exist_ok=True)
            if self.use_tensorboard:    # init the tensorboard writer (SummaryWriter):
                tensorboard_dir = os.path.join(self.logdir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.tb_writer = tb.SummaryWriter(log_dir=tensorboard_dir)

        return

    def show(self, log, prompt: str = ""):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            print("%s%s" % (prompt, log))
        else:
            pass
        return

    def print_config(self, config: dict, prompt: str = ""):
        if self.is_activate:
            print(prompt, end="")
            for _ in config:
                print(f"{_.lower()}:{config[_]}; ", end="")
            print("")

    def save_config(self, config: dict, filename: str):
        if self.is_activate:
            self._write_dict_to_yaml(x=config, filename=filename, mode="w")
        return

    def print_metrics(self, metrics: Metrics, prompt: str = "", fmt: str = "{average:.4f} ({global_average:.4f})"):
        if self.is_activate:
            print(prompt, end="")
            print(metrics.fmt(fmt=fmt))
        return

    def save_metrics(self, metrics: Metrics, prompt: str = "",
                     fmt: str = "{average:.4f} ({global_average:.4f})", statistic: str = "average"):
        """

        Args:
            metrics:
            prompt:
            fmt:
            statistic:

        Returns:

        """

    def print(self, log):
        pass

    def write(self, log, filename: str, mode: str = "w"):
        """
        Logger write a log to a file.

        Args:
            log: A log.
            filename: Write file name.
            mode: Open file with this mode.
        """
        if (self.only_main and is_main_process()) or (self.only_main is False):
            if isinstance(log, dict):
                if len(filename) > 5 and filename[-5:] == ".yaml":
                    self.write_dict_to_yaml(log, filename, mode)
                elif len(filename) > 5 and filename[-5:] == ".json":
                    self.write_dict_to_json(log, filename, mode)
                else:
                    raise RuntimeError("Filename '%s' is not supported for dict log." % filename)
            elif isinstance(log, MetricLo):
                with open(os.path.join(self.logdir, filename), mode=mode) as f:
                    f.write(log.__str__() + "\n")
            else:
                raise RuntimeError("Log type '%s' is not supported." % type(log))
        else:
            pass
        return

    def _write_dict_to_yaml(self, x: dict, filename: str, mode: str = "w"):
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(x, f, allow_unicode=True)
        return

    def write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return

    def tb_add_scalars(self, main_tag: str, tag_scalar_dict: dict, global_step: int):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tb_logger.add_scalars(
                main_tag=main_tag,
                tag_scalar_dict=tag_scalar_dict,
                global_step=global_step
            )
        else:
            pass
        return

    def tb_add_scalar(self, tag: str, scalar_value: float, global_step: int):
        if (self.only_main and is_main_process()) or (self.only_main is False):
            self.tb_logger.add_scalar(
                tag=tag,
                scalar_value=scalar_value,
                global_step=global_step
            )
        else:
            pass
        return


def parser_to_dict(log: argparse.ArgumentParser) -> dict:
    """
    Transform options to a dict.

    Args:
        log: The options.

    Returns:
        Options dict.
    """
    opts_dict = dict()
    for k, v in vars(log).items():
        if v:
            opts_dict[k] = v
    return opts_dict
