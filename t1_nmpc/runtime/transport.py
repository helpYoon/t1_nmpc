"""Transport: the read-state / write-command boundary shared by the MuJoCo sim and the robot SDK,
so the same threaded control loop drives either."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..robot.config import JointCommand


class Transport(Protocol):
    def read_state(self) -> np.ndarray:        # 68-d WB state x
        ...

    def write_command(self, cmd: JointCommand) -> None:
        ...

    def now(self) -> float:                    # monotonic time (sim clock or wall)
        ...
