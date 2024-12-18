from __future__ import annotations

import logging
import sys
from pathlib import Path

from . import distributed

__all__ = ["logger", "log_to_file", "set_log_level"]


class MultiLineFormatter(logging.Formatter):
    """Detect multi-line logs and adds newlines with colors."""

    def __init__(self):
        super().__init__("[%(name)s %(levelname)s]: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        record.msg = str(record.msg).strip()
        # add in new lines
        if "\n" in record.msg:
            record.msg = "\n" + record.msg + "\n"
        return super().format(record)


# create the graph-pes logger
logger = logging.getLogger(name="graph-pes")
std_out_handler = logging.StreamHandler(stream=sys.stdout)
std_out_handler.setFormatter(MultiLineFormatter())

# log to stdout if rank 0
if distributed.IS_RANK_0:
    logger.addHandler(std_out_handler)

    # capture all logs but only show INFO and above in stdout (by default)
    logger.setLevel(logging.DEBUG)
    std_out_handler.setLevel(logging.INFO)


def log_to_file(output_dir: str | Path):
    """Append logs to a `rank-<rank>.log` file in the given output directory."""

    file = Path(output_dir) / f"rank-{distributed.GLOBAL_RANK}.log"
    file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(file, mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MultiLineFormatter())

    logger.addHandler(handler)
    logger.info(f"Logging to {file}")


def set_log_level(level: str | int):
    """Set the logging level."""

    std_out_handler.setLevel(level)
    logger.debug(f"Set logging level to {level}")
