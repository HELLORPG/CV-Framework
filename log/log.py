# @Author       : Ruopeng Gao
# @Date         : 2022/7/13
# @Description  :
import torch.distributed

from typing import List, Any
from utils.utils import is_distributed, distributed_world_size


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
