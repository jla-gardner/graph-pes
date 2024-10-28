from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["logger", "log_to_file", "set_level"]


class MultiLineFormatter(logging.Formatter):
    """Detect multi-line logs and adds newlines."""

    def __init__(self):
        super().__init__("[%(name)s %(levelname)s]: %(message)s")

    def format(self, record):
        record.msg = str(record.msg).strip()
        if "\n" in record.msg:
            record.msg = "\n" + record.msg + "\n"
        return super().format(record)


# create the graph-pes logger
logger = logging.getLogger(name="graph-pes")

# log to stdout
_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(MultiLineFormatter())
logger.addHandler(_handler)

# capture all logs but only show INFO and above in stdout (by default)
logger.setLevel(logging.DEBUG)
_handler.setLevel(logging.INFO)


def log_to_file(file: str | Path):
    """Append logs to a file."""

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(file, mode="a")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(MultiLineFormatter())

    logger.addHandler(handler)
    logger.info(f"Logging to {file}")


def set_level(level: str | int):
    """Set the logging level."""

    _handler.setLevel(level)
    logger.info(f"Set logging level to {level}")
