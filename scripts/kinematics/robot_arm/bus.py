"""Bus série Feetech : connexion, lecture/écriture des ticks, interpolation.

Les positions sont écrites comme move_central_position.py : Goal_Position en
ticks bruts, normalize=False, via sync_write.
Le couple reste actif au débranchement (disable_torque=False).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "robot"))

from motors_config import MOTORS  # noqa: E402
from safety import write_safe  # noqa: E402

from .calibration import ALL_MOTOR_NAMES  # noqa: E402

DEFAULT_SETTLE_S = 2.5
DEFAULT_MOVE_S = 2.0
RATE_HZ = 20


@contextmanager
def connect_arm(port: str):
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port=port, motors=MOTORS)
    bus.connect()
    try:
        yield bus
    finally:
        bus.disconnect(disable_torque=False)


def read_present_raw(bus) -> dict[str, int]:
    return {n: int(v) for n, v in bus.sync_read("Present_Position", ALL_MOTOR_NAMES, normalize=False).items()}


def write_goal_raw(bus, raw: dict[str, int]) -> None:
    write_safe(bus, list(raw), list(raw.values()), normalize=False)


def interpolate_raw(start: dict[str, int], end: dict[str, int], move_s: float) -> list[dict[str, int]]:
    """Consignes intermédiaires linéaires de start à end, une par période de RATE_HZ (end inclus)."""
    n = max(1, round(move_s * RATE_HZ))
    return [{m: round(start[m] + (end[m] - start[m]) * k / n) for m in end} for k in range(1, n + 1)]
