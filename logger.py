# @Author       : Ruopeng Gao
# @Date         : 2022/7/5
# @Description  : Logger will log information.


import os
import json


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

    def log_dict(self, log: dict, filename: str, mode: str = "w+"):
        """
        Log a dict data.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w+" or "a+".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return



