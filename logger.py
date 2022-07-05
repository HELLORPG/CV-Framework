# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.


import os
import json
import argparse
import yaml


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



