"""Calibration et conversions entre ticks servos et degrés Lerobot.

Conversion ticks <-> degrés : même logique que Lerobot (SOFollower,
use_degrees=True, MotorNormMode.DEGREES), qui est aussi celle attendue par
l'URDF so101_new_calib et par RobotKinematics :
    degrés = (Present_Position - milieu) * 360 / 4095,  milieu = (range_min + range_max) / 2
Present_Position est déjà corrigée par le servo (Homing_Offset écrit en EEPROM
pendant la calibration), donc homing_offset n'intervient pas dans le calcul.
Le gripper suit RANGE_0_100 : 0 = range_min, 100 = range_max.

Source de la calibration : les registres des servos (Homing_Offset,
Min/Max_Position_Limit, lus comme bus.read_calibration() de Lerobot) dès qu'un
port est donné, car c'est avec eux que le servo calcule Present_Position. Sans
port (aperçu dry-run), le fichier .cache/.../follower_arm.json est utilisé ; s'il
diffère des servos, un avertissement est affiché.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CALIB_FILE = (
    REPO_ROOT / ".cache" / "huggingface" / "lerobot" / "calibration"
    / "robots" / "so_follower" / "follower_arm.json"
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_MOTOR_NAMES = JOINT_NAMES + ["gripper"]

# STS3215 : résolution 4096, Lerobot divise par max_res = 4095.
MAX_RES = 4095
GRIPPER_DEFAULT = 50.0


def load_calibration_file(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {n: {k: int(data[n][k]) for k in ("homing_offset", "range_min", "range_max")} for n in ALL_MOTOR_NAMES}


def read_servo_calibration(bus) -> dict[str, dict]:
    """Registres de calibration des servos, comme FeetechMotorsBus.read_calibration()."""
    return {
        n: {"homing_offset": c.homing_offset, "range_min": c.range_min, "range_max": c.range_max}
        for n, c in bus.read_calibration().items()
    }


def calibration_mismatches(servo: dict, file: dict) -> list[str]:
    return [n for n in ALL_MOTOR_NAMES if servo[n] != file[n]]


def resolve_calibration(port: str | None, calib_file: Path) -> dict[str, dict]:
    """Registres des servos si un port est donné, sinon le fichier (aperçu)."""
    file_calib = load_calibration_file(calib_file) if calib_file.exists() else None

    if port is None:
        if file_calib is None:
            sys.exit(f"Ni port ni fichier de calibration ({calib_file}).")
        print(f"[calib] fichier {calib_file} (aperçu sans port : peut différer des servos)")
        return file_calib

    from .bus import connect_arm  # différé : bus importe les noms de ce module

    with connect_arm(port) as bus:
        servo_calib = read_servo_calibration(bus)
    print(f"[calib] registres des servos sur {port}")
    if file_calib is not None:
        diff = calibration_mismatches(servo_calib, file_calib)
        if diff:
            print(f"[calib] ATTENTION : {calib_file.name} ne correspond pas aux servos ({', '.join(diff)}).\n"
                  "         Lerobot (teleop/record) proposera d'écrire ce fichier dans les servos : "
                  "cela changerait leur repère.")
    return servo_calib


# ---------------------------------------------------------------------------
# Conversions (identiques à lerobot MotorsBus._normalize / _unnormalize)
# ---------------------------------------------------------------------------

def _mid(c: dict) -> float:
    return (c["range_min"] + c["range_max"]) / 2


def raw_to_deg(name: str, raw: float, calib: dict) -> float:
    return (raw - _mid(calib[name])) * 360 / MAX_RES


def deg_to_raw(name: str, deg: float, calib: dict) -> tuple[int, bool]:
    """Degrés -> ticks, bornés à la plage calibrée. Renvoie (ticks, borné ?)."""
    c = calib[name]
    raw = int(deg * MAX_RES / 360 + _mid(c))
    clipped = min(c["range_max"], max(c["range_min"], raw))
    return clipped, clipped != raw


def deg_limits(name: str, calib: dict) -> tuple[float, float]:
    c = calib[name]
    return raw_to_deg(name, c["range_min"], calib), raw_to_deg(name, c["range_max"], calib)


def gripper_pct_to_raw(pct: float, calib: dict) -> int:
    c = calib["gripper"]
    return int(min(100.0, max(0.0, pct)) / 100 * (c["range_max"] - c["range_min"]) + c["range_min"])


def gripper_raw_to_pct(raw: float, calib: dict) -> float:
    c = calib["gripper"]
    return (raw - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100


# ---------------------------------------------------------------------------
# Pose = 5 angles (degrés) + gripper (%)
# ---------------------------------------------------------------------------

def pose_from_raw(raw: dict[str, int], calib: dict) -> dict[str, float]:
    pose = {n: raw_to_deg(n, raw[n], calib) for n in JOINT_NAMES}
    pose["gripper"] = gripper_raw_to_pct(raw["gripper"], calib)
    return pose


def pose_to_raw(pose: dict[str, float], calib: dict) -> tuple[dict[str, int], list[str]]:
    raw, clipped = {}, []
    for n in JOINT_NAMES:
        raw[n], was_clipped = deg_to_raw(n, pose[n], calib)
        if was_clipped:
            clipped.append(n)
    raw["gripper"] = gripper_pct_to_raw(pose["gripper"], calib)
    return raw, clipped


def neutral_pose() -> dict[str, float]:
    pose = {n: 0.0 for n in JOINT_NAMES}
    pose["gripper"] = GRIPPER_DEFAULT
    return pose
