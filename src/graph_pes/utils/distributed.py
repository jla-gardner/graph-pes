from __future__ import annotations

import time
from pathlib import Path
from typing import Final

from graph_pes.utils.misc import silently_create_trainer

# dirty hack: just get lightning to work this out,
# and ensure no annoying printing happens
_trainer = silently_create_trainer(logger=False)

GLOBAL_RANK: Final[int] = _trainer.global_rank
WORLD_SIZE: Final[int] = _trainer.world_size
IS_RANK_0: Final[bool] = GLOBAL_RANK == 0


class Communication:
    def __init__(self, key: str, timeout_s: float = 20.0):
        if "\n" in key:
            raise ValueError("Key cannot contain newlines")
        self.key = key
        self.timeout_s = timeout_s
        self.path = Path(self.key)
        assert not self.path.exists()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # wait for all ranks to reach this point
        # so that the path and contents are not lost
        # before other ranks have read them
        _trainer.strategy.barrier(self.key)
        self.path.unlink(missing_ok=True)

    def send_to_other_ranks(self, value: str) -> None:
        assert IS_RANK_0

        # write the value to the path on rank 0
        self.path.write_text(value)

    def receive_from_rank_0(self) -> str:
        assert not IS_RANK_0

        # read the value from the path on non-0 ranks,
        # re-trying a few times until the file is found
        # in cases where rank 0 is slow

        for _ in range(int(self.timeout_s * 10)):
            if self.path.exists():
                return self.path.read_text()
            time.sleep(0.1)

        raise RuntimeError(
            f"Key {self.key} not found after {self.timeout_s}s "
            f"on rank {GLOBAL_RANK}"
        )
