# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.
import os
import json
import argparse
import yaml
import torch.distributed

from typing import List, Any
from torch.utils import tensorboard as tb
from utils.utils import is_distributed, distributed_world_size, is_main_process


class MetricLog:
    def __init__(self, epoch: int = None):
        """

        Args:
            epoch: Current Epoch.
        """
        self.epoch = epoch
        self.metrics = dict()   # The metric value.
        self.counts = dict()     # The count.
        self.mean_metrics = dict()

    def update(self, metric_name: str, metric_value: float, count: int):
        """

        Args:
            metric_name:
            metric_value:
            count:
        """
        if metric_name not in self.metrics.keys():
            self.metrics[metric_name] = list()
            self.counts[metric_name] = list()
        assert isinstance(self.metrics[metric_name], list)
        self.metrics[metric_name].append(metric_value)
        self.counts[metric_name].append(count)
        return

    def mean(self):
        self.mean_metrics = dict()
        if is_distributed():
            # 如果分布式训练，则需要计算所有节点上的平均值。
            torch.distributed.barrier()
            metrics_gather = [None] * distributed_world_size()
            counts_gather = [None] * distributed_world_size()
            torch.distributed.all_gather_object(metrics_gather, self.metrics)
            torch.distributed.all_gather_object(counts_gather, self.counts)
            metrics_gather, counts_gather = merge_dicts(metrics_gather), merge_dicts(counts_gather)
            for k in metrics_gather.keys():
                self.mean_metrics[k] = \
                    sum([i*j for i, j in zip(metrics_gather[k], counts_gather[k])]) / sum(counts_gather[k])
        else:
            for k in self.metrics.keys():
                self.mean_metrics[k] = sum([i*j for i, j in zip(self.metrics[k], self.counts[k])]) / sum(self.counts[k])
        return

    def __str__(self):
        s = str()
        if self.epoch is not None:
            s += "Epoch %d: " % self.epoch
        s += self.mean_metrics.__str__()
        return s

    @classmethod
    def concat(cls, metrics: List[Any]):
        """
        Concat different metrics.

        Args:
            metrics:

        Returns:
            Concat result.
        """
        log = MetricLog()
        for m in metrics:
            if m.epoch is not None:
                log.epoch = m.epoch
            log.metrics.update(m.metrics)
            log.counts.update(m.counts)
            log.mean_metrics.update(m.mean_metrics)
        return log


class ProgressLog:
    def __init__(self, epoch: int = None, total_step: int = None):
        self.epoch = epoch
        self.total_step = total_step
        self.current_step = None

    def update(self, current_step: int):
        self.current_step = current_step

    def fraction(self):
        return self.current_step / self.total_step


class Logger:
    """
    Log information.
    """
    def __init__(self, logdir: str):
        """
        Create a logger.

        Args:
            logdir (str): Logger outputs path.
        """
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(os.path.join(self.logdir, "tb_log"), exist_ok=True)
        self.tb_logger = tb.SummaryWriter(log_dir=os.path.join(self.logdir, "tb_log"))
        return

    @classmethod
    def show(cls, log, prompt: str = ""):
        print("%s%s" % (prompt, log))
        return

    def write(self, log, filename: str, mode: str = "w"):
        """
        Logger write a log to a file.

        Args:
            log: A log.
            filename: Write file name.
            mode: Open file with this mode.
        """
        if isinstance(log, dict):
            if len(filename) > 5 and filename[-5:] == ".yaml":
                self.write_dict_to_yaml(log, filename, mode)
            elif len(filename) > 5 and filename[-5:] == ".json":
                self.write_dict_to_json(log, filename, mode)
            else:
                raise RuntimeError("Filename '%s' is not supported for dict log." % filename)
        elif isinstance(log, MetricLog):
            with open(os.path.join(self.logdir, filename), mode=mode) as f:
                f.write(log.__str__() + "\n")
        else:
            raise RuntimeError("Log type '%s' is not supported." % type(log))

    def write_dict_to_yaml(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .yaml file.

        Args:
            log: A dict log.
            filename: A yaml file's name.
            mode: Open with this mode.
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            yaml.dump(log, f, allow_unicode=True)
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
        self.tb_logger.add_scalars(
            main_tag=main_tag,
            tag_scalar_dict=tag_scalar_dict,
            global_step=global_step
        )

    def tb_add_scalar(self, tag: str, scalar_value: float, global_step: int):
        self.tb_logger.add_scalar(
            tag=tag,
            scalar_value=scalar_value,
            global_step=global_step
        )


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


def merge_dicts(dicts: List[dict]) -> dict:
    """
    将输入的两个字典进行合并，字典的值均是 list 形式。
    所有 list 拼接成为一个最终的 list 作为最终的值。

    Args:
        dicts:

    Returns:
        Merged dict.
    """
    merged = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in merged.keys():
                merged[k] = list()
            merged[k] += v
    return merged


