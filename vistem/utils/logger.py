import os
import sys
import functools
import logging
import time
from termcolor import colored

from vistem.dist.get_info import get_rank
from .caller import find_caller

__all__ = ['setup_logger']

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name")
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        prefix = time.strftime('[%m/%d %H:%M:%S]', time.localtime(time.time()))
        prefix = f'{prefix} {self._root_name}'

        if record.levelno == logging.DEBUG:
            prefix = colored(prefix, "blue")
        elif record.levelno == logging.INFO:
            prefix = colored(prefix, "green")
        elif record.levelno == logging.WARNING:
            prefix = colored(prefix, "yellow")
        elif record.levelno == logging.ERROR:
            prefix = colored(prefix, "red", attrs=["blink"])
        elif record.levelno == logging.CRITICAL:
            prefix = colored(prefix, "red", attrs=["blink", "underline"])
        return prefix + " " + log

@functools.lru_cache()
def setup_logger(name=None, output=None):
    caller = find_caller()['caller']
    logger = logging.getLogger(caller)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # stdout logging: master only
    if get_rank() == 0:
        formatter = _ColorfulFormatter("%(message)s", root_name=caller)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log") :
            filename = output
        else :
            if not os.path.isdir(output) : os.makedirs(output)
            filename = os.path.join(output, "log.txt")

        if get_rank() > 0:
            filename = f'{filename}.rank{get_rank()}'

        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(message)s", datefmt="%m/%d %H:%M:%S"
        )
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")

