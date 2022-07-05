# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.


import os
import json
import argparse


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
        return

    @classmethod
    def log(cls, log, prompt: str = ""):
        print("%s%s" % (prompt, log))
        return


    def log_dict_to_file(self, log: dict, filename: str, mode: str = "w"):
        """
        Log a dict data.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
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



