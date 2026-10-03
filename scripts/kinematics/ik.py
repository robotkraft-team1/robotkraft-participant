"""Cinématique inverse pour le SO-ARM101 : donner un point (ou des angles), le bras s'y rend.

Les positions sont écrites comme move_central_position.py : Goal_Position en
ticks bruts, normalize=False, via sync_write.

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

Commandes :
    --point X Y Z      point en mètres, repère base du bras, résolu par IK ;
    --joints J=DEG ... angles absolus en degrés (0 = milieu de la plage calibrée) ;
    --sequence FICHIER liste d'étapes exécutées dans l'ordre (voir sequence_points.json) ;
    --status           lecture seule de la position courante (angles + x, y, z de l'outil).

Repère base (celui de l'URDF) : origine sous le socle, sur la table ; x vers l'avant
du bras quand shoulder_pan est au milieu de sa plage, y vers sa gauche, z vers le haut.
Le point amené en (x, y, z) est celui de l'outil choisi (--tool, ou "tool" dans la
séquence, ou le "default" de tools.json) : un repère défini dans tools.json par son
décalage depuis gripper_frame_link (face intérieure de la mâchoire fixe, 6 mm avant
son bout).

Orientation (optionnelle, en degrés : --yaw/--pitch/--roll, ou "orientation" dans la
séquence), donnée pour l'axe de la pince (l'axe de wrist_roll, qui sort de la pince) :
    pitch : inclinaison de cet axe, 0 = horizontal, -90 = pince vers le bas ;
    yaw   : direction où il pointe vu du dessus, 0 = +x, 90 = +y. Le bras n'a que
            5 axes : le yaw est imposé par la position (il vaut -shoulder_pan). Il
            n'est donc pas commandé, seulement vérifié s'il est donné ;
    roll  : rotation autour de cet axe, positive dans le sens horaire vu depuis le
            poignet (roll = -wrist_roll) ; 0 = wrist_roll au milieu de sa plage.
Sans orientation, l'IK contraint la position seule et wrist_roll reste inchangé ; un
angle omis reste libre (pitch) ou inchangé (roll).

Avant toute exécution, le plan est refusé si un point (avec son orientation) est hors
de portée, si le yaw demandé diffère de plus de 5° de celui imposé par la position, si
une cible est en butée, ou si le gripper ou l'outil passe sous --min-z (cibles et trajets).

DRY-RUN par défaut. --execute écrit sur le bus ; chaque cible est rejointe par
interpolation linéaire en --move-time secondes (comme visualize.py), et le couple
reste actif au débranchement (disable_torque=False).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "robot"))

from motors_config import MOTORS  # noqa: E402
from safety import write_safe  # noqa: E402

URDF_PATH = REPO_ROOT / "urdf" / "so101" / "so101_new_calib.urdf"
TOOLS_FILE = Path(__file__).resolve().parent / "tools.json"
DEFAULT_CALIB_FILE = (
    REPO_ROOT / ".cache" / "huggingface" / "lerobot" / "calibration"
    / "robots" / "so_follower" / "follower_arm.json"
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
ALL_MOTOR_NAMES = JOINT_NAMES + ["gripper"]

# STS3215 : résolution 4096, Lerobot divise par max_res = 4095.
MAX_RES = 4095
GRIPPER_DEFAULT = 50.0
DEFAULT_SETTLE_S = 2.5
DEFAULT_MOVE_S = 2.0
RATE_HZ = 20

# RobotKinematics.inverse_kinematics ne fait qu'un pas de solveur (Lerobot l'appelle
# à chaque cycle de contrôle) : on itère jusqu'à convergence.
IK_MAX_ITERS = 200
IK_CONVERGED_M = 0.0005
IK_CONVERGED_DEG = 0.05
IK_TOLERANCE_M = 0.005   # au-delà, le point est jugé hors de portée
IK_TOLERANCE_DEG = 1.0   # au-delà, l'orientation (pitch, roll) est jugée inatteignable
YAW_TOLERANCE_DEG = 5.0  # écart toléré entre un yaw demandé et celui imposé par la position
ORIENTATION_WEIGHT = 0.5

# La table est à z = 0 dans le repère base (dessous du socle de l'URDF).
DEFAULT_MIN_Z_M = 0.01
# Boîte englobant les meshes du gripper (moteur de roll et mâchoires inclus),
# dans le repère gripper_frame_link, en m.
GRIPPER_BOX = np.array([[x, y, z] for x in (-0.038, 0.027) for y in (-0.028, 0.024) for z in (-0.099, 0.007)])
PATH_SAMPLES = 40


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

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
# Cinématique inverse (placo + URDF so101_new_calib)
# ---------------------------------------------------------------------------

def load_tools(path: Path = TOOLS_FILE) -> tuple[dict[str, dict], str]:
    """Outils de tools.json : {nom: {"xyz": [m], "rpy": [rad]}} et le nom de l'outil par défaut."""
    with open(path) as f:
        data = json.load(f)
    tools = {
        name: {"xyz": [float(v) for v in t["xyz_m"]], "rpy": [float(np.deg2rad(v)) for v in t["rpy_deg"]]}
        for name, t in data["tools"].items()
    }
    if data["default"] not in tools:
        raise ValueError(f"{path.name} : outil par défaut {data['default']!r} absent de 'tools'")
    return tools, data["default"]


def rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle de la rotation qui amène b sur a."""
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1.0, 1.0))))


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def tool_matrix(tool: dict) -> np.ndarray:
    """Placement de l'outil dans gripper_frame_link (rpy fixes comme dans l'URDF : Rz·Ry·Rx)."""
    roll, pitch, yaw = np.degrees(tool["rpy"])
    t = np.eye(4)
    t[:3, :3] = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    t[:3, 3] = tool["xyz"]
    return t


@contextmanager
def urdf_with_tools(tools: dict[str, dict]):
    """Copie temporaire de l'URDF : meshes en chemin absolu, un repère fixe 'tcp_<nom>' par outil."""
    text = re.sub(r'filename="assets/', f'filename="{URDF_PATH.parent / "assets"}/', URDF_PATH.read_text())
    frames = "".join(
        f'<link name="tcp_{name}"/><joint name="tcp_{name}_joint" type="fixed">'
        f'<origin xyz="{" ".join(map(str, t["xyz"]))}" rpy="{" ".join(map(str, t["rpy"]))}"/>'
        f'<parent link="gripper_frame_link"/><child link="tcp_{name}"/></joint>'
        for name, t in tools.items()
    )
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(text.replace("</robot>", frames + "</robot>"))
    try:
        yield Path(f.name)
    finally:
        os.unlink(f.name)


class ArmKinematics:
    """IK/FK (placo + URDF so101_new_calib) pour le point d'un outil de tools.json."""

    def __init__(self, tool: str | None = None, tools_file: Path = TOOLS_FILE):
        from lerobot.model.kinematics import RobotKinematics

        tools, default = load_tools(tools_file)
        self.tool = tool or default
        if self.tool not in tools:
            sys.exit(f"Outil inconnu : {self.tool!r} (dans {tools_file.name} : {', '.join(tools)})")
        self.offset = tool_matrix(tools[self.tool])
        with urdf_with_tools({self.tool: tools[self.tool]}) as urdf:
            self._kin = RobotKinematics(urdf_path=str(urdf), target_frame_name=f"tcp_{self.tool}",
                                        joint_names=JOINT_NAMES)
            self._kin_with_roll = RobotKinematics(urdf_path=str(urdf), target_frame_name=f"tcp_{self.tool}",
                                                  joint_names=JOINT_NAMES)
        # Sans roll imposé, le roll ne déplace presque pas le point visé : libre, le solveur le fait dériver.
        self._kin.solver.mask_dof("wrist_roll")

        # La convention yaw/pitch/roll suppose que l'axe z de l'outil est l'axe de la pince (celui du roll).
        self.supports_orientation = bool(np.allclose(self.offset[:3, 2], [0.0, 0.0, 1.0], atol=1e-6))
        # Référence du roll : orientation de l'outil quand tous les joints sont au milieu de leur plage.
        zero = self.tool_transform(dict.fromkeys(JOINT_NAMES, 0.0))[:3, :3]
        self._roll_ref = rot_y(np.degrees(np.arctan2(zero[2, 2], zero[0, 2]))) @ zero

    def tool_transform(self, pose: dict[str, float]) -> np.ndarray:
        """FK : pose (degrés Lerobot) -> matrice 4x4 de l'outil dans le repère base."""
        return self._kin.forward_kinematics(np.array([pose[n] for n in JOINT_NAMES], dtype=float))

    def rotation(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Orientation de l'outil (degrés, convention en tête de fichier) -> matrice 3x3 dans le repère base."""
        return rot_z(yaw) @ rot_y(-pitch) @ self._roll_ref @ rot_z(roll)

    def orientation(self, pose: dict[str, float]) -> tuple[float, float, float]:
        """(yaw, pitch, roll) de l'outil en degrés. L'axe de la pince reste dans le plan vertical
        du bras, dont l'azimut est -shoulder_pan (l'axe du pan pointe vers le bas dans l'URDF)."""
        yaw = -pose["shoulder_pan"]
        in_plane = rot_z(-yaw) @ self.tool_transform(pose)[:3, :3]
        pitch = float(np.degrees(np.arctan2(in_plane[2, 2], in_plane[0, 2])))
        about_axis = self._roll_ref.T @ rot_y(pitch) @ in_plane
        roll = float(np.degrees(np.arctan2(about_axis[1, 0], about_axis[0, 0])))
        return yaw, pitch, roll

    def gripper_min_z(self, pose: dict[str, float]) -> float:
        """Point le plus bas du gripper (sa boîte englobante) et de l'outil."""
        gripper = self.tool_transform(pose) @ np.linalg.inv(self.offset)
        points = np.vstack([GRIPPER_BOX, self.offset[:3, 3]])
        return float((points @ gripper[:3, :3].T + gripper[:3, 3])[:, 2].min())

    def ik_point(self, x: float, y: float, z: float, guess_deg: dict[str, float],
                 pitch: float | None = None, roll: float | None = None) -> dict[str, float]:
        """IK : point (m, repère base), et pitch/roll optionnels (degrés) -> angles (degrés Lerobot).

        Le yaw suit toujours le plan du bras. Sans roll imposé, wrist_roll reste inchangé ;
        sans pitch imposé, l'inclinaison est libre.
        """
        if (pitch is not None or roll is not None) and not self.supports_orientation:
            sys.exit(f"Outil {self.tool!r} : son axe z n'est pas celui de la pince, orientation non supportée")
        kin = self._kin_with_roll if roll is not None else self._kin
        weight = 0.0 if pitch is None and roll is None else ORIENTATION_WEIGHT
        target = np.eye(4)
        target[:3, 3] = [x, y, z]
        q = np.array([guess_deg[n] for n in JOINT_NAMES], dtype=float)
        for _ in range(IK_MAX_ITERS):
            if weight:
                # Le yaw de la cible suit le pan courant : il est recalculé à chaque pas.
                yaw, current_pitch, current_roll = self.orientation(dict(zip(JOINT_NAMES, q)))
                target[:3, :3] = self.rotation(yaw, current_pitch if pitch is None else pitch,
                                               current_roll if roll is None else roll)
            q = kin.inverse_kinematics(current_joint_pos=q, desired_ee_pose=target, orientation_weight=weight)
            pose = dict(zip(JOINT_NAMES, q))
            if (np.linalg.norm(self.tool_transform(pose)[:3, 3] - target[:3, 3]) < IK_CONVERGED_M
                    and self.orientation_error(pose, pitch, roll) < IK_CONVERGED_DEG):
                break
        return {n: float(q[i]) for i, n in enumerate(JOINT_NAMES)}

    def orientation_error(self, pose: dict[str, float], pitch: float | None, roll: float | None) -> float:
        """Plus grand écart (degrés) entre le pitch/roll atteints et ceux demandés (0 si rien n'est demandé)."""
        _, reached_pitch, reached_roll = self.orientation(pose)
        errors = [abs(wrap_deg(reached_pitch - pitch)) if pitch is not None else 0.0,
                  abs(wrap_deg(reached_roll - roll)) if roll is not None else 0.0]
        return max(errors)


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

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


def print_pose(raw: dict[str, int], calib: dict, clipped: list[str] = ()) -> None:
    print(f"{'Joint':<15} {'deg':>8} {'ticks':>6}   {'plage deg':>17}")
    for n in JOINT_NAMES:
        lo, hi = deg_limits(n, calib)
        flag = "  <- butée" if n in clipped else ""
        print(f"{n:<15} {raw_to_deg(n, raw[n], calib):>+8.1f} {raw[n]:>6}   [{lo:>+6.1f}, {hi:>+6.1f}]{flag}")
    print(f"{'gripper':<15} {gripper_raw_to_pct(raw['gripper'], calib):>7.0f}% {raw['gripper']:>6}")


# ---------------------------------------------------------------------------
# Séquence
# ---------------------------------------------------------------------------

def load_sequence(path: Path) -> tuple[list[dict], str | None]:
    """Étapes de la séquence et outil qu'elle déclare ("tool", optionnel)."""
    with open(path) as f:
        data = json.load(f)
    steps = data["steps"] if isinstance(data, dict) else data
    if not steps:
        sys.exit(f"Séquence vide : {path}")
    return steps, data.get("tool") if isinstance(data, dict) else None


def plan_sequence(steps: list[dict], calib: dict, initial: dict[str, float], kin: ArmKinematics) -> list[dict]:
    """Ticks cibles de chaque étape, la pose courante étant portée d'une étape à l'autre.

    Étape : {"return": "initial"} |
            {"point": [x, y, z], "orientation": {"yaw": deg, "pitch": deg, "roll": deg}} (chacun optionnel) |
            {"joints": {nom: deg}, "relative": bool}, avec "gripper" (%) optionnel.
    """
    pose = dict(initial)
    planned = []
    for i, step in enumerate(steps):
        label = step.get("label", f"étape {i + 1}")
        target, wanted = None, {}
        if "return" in step:
            pose = dict(initial)
            desc = "retour à la position de départ"
        elif "point" in step:
            x, y, z = target = [float(v) for v in step["point"]]
            wanted = {k: float(v) for k, v in step.get("orientation", {}).items()}
            if set(wanted) - {"yaw", "pitch", "roll"}:
                sys.exit(f"Étape {i + 1} : orientation attendue en yaw, pitch, roll (reçu {', '.join(wanted)})")
            pose.update(kin.ik_point(x, y, z, pose, wanted.get("pitch"), wanted.get("roll")))
            desc = f"point base ({x:.3f}, {y:.3f}, {z:.3f}) m"
            if wanted:
                desc += ", " + ", ".join(f"{k}={v:g}°" for k, v in wanted.items())
        elif "joints" in step:
            for n, v in step["joints"].items():
                if n not in JOINT_NAMES:
                    sys.exit(f"Joint inconnu dans la séquence : {n!r} ({', '.join(JOINT_NAMES)})")
                pose[n] = pose[n] + float(v) if step.get("relative") else float(v)
            sign = "+" if step.get("relative") else ""
            desc = "joints " + ", ".join(f"{n}={sign if v >= 0 else ''}{v:g}°" for n, v in step["joints"].items())
        else:
            sys.exit(f"Étape {i + 1} : 'point', 'joints' ou 'return' requis")
        if "gripper" in step:
            pose["gripper"] = float(step["gripper"])

        raw, clipped = pose_to_raw(pose, calib)
        pose = pose_from_raw(raw, calib)  # la pose suivante part de la cible réellement atteignable
        ee = kin.tool_transform(pose)[:3, 3]
        planned.append({
            "label": label, "desc": desc, "raw": raw, "clipped": clipped, "settle_s": step.get("settle_s"),
            "ee": ee, "ik_error": None if target is None else float(np.linalg.norm(ee - target)),
            "target": target, "wanted": wanted, "target_rotation": target_rotation(kin, pose, wanted),
            "orientation_error": kin.orientation_error(pose, wanted.get("pitch"), wanted.get("roll")),
        })
    return planned


def target_rotation(kin: ArmKinematics, pose: dict[str, float], wanted: dict[str, float]) -> np.ndarray | None:
    """Rotation demandée ; le yaw et les angles non précisés sont ceux de la pose atteinte."""
    if not wanted:
        return None
    yaw, pitch, roll = kin.orientation(pose)
    return kin.rotation(yaw, wanted.get("pitch", pitch), wanted.get("roll", roll))


def path_min_z(start_raw: dict[str, int], end_raw: dict[str, int], calib: dict, kin: ArmKinematics) -> float:
    """Point le plus bas du gripper le long du trajet interpolé (cible incluse)."""
    return min(
        kin.gripper_min_z(pose_from_raw({m: start_raw[m] + (end_raw[m] - start_raw[m]) * k / PATH_SAMPLES
                                         for m in end_raw}, calib))
        for k in range(1, PATH_SAMPLES + 1)
    )


def check_plan(initial_raw: dict[str, int], planned: list[dict], calib: dict, min_z: float,
               kin: ArmKinematics) -> list[str]:
    """Ajoute 'path_min_z' à chaque étape ; renvoie les problèmes qui interdisent l'exécution."""
    problems = []
    current = initial_raw
    for step in planned:
        step["path_min_z"] = path_min_z(current, step["raw"], calib, kin)
        current = step["raw"]
        wanted = step["wanted"]
        missed = step["ik_error"] is not None and step["ik_error"] > IK_TOLERANCE_M
        if wanted and (missed or step["orientation_error"] > IK_TOLERANCE_DEG):
            problems.append(f"{step['label']} : point et orientation inatteignables ensemble (manqué de "
                            f"{step['ik_error'] * 1000:.0f} mm, {step['orientation_error']:.1f}°)")
        elif missed:
            problems.append(f"{step['label']} : point hors de portée (manqué de {step['ik_error'] * 1000:.0f} mm)")
        if "yaw" in wanted:
            imposed = kin.orientation(pose_from_raw(step["raw"], calib))[0]
            if abs(wrap_deg(wanted["yaw"] - imposed)) > YAW_TOLERANCE_DEG:
                problems.append(f"{step['label']} : yaw {wanted['yaw']:g}° incompatible, "
                                f"la position impose yaw = {imposed:.1f}°")
        if step["clipped"]:
            problems.append(f"{step['label']} : butée atteinte ({', '.join(step['clipped'])})")
        if step["path_min_z"] < min_z:
            problems.append(f"{step['label']} : le gripper descend à z = {step['path_min_z'] * 1000:.0f} mm "
                            f"(minimum {min_z * 1000:.0f} mm)")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_joints(pairs: list[str]) -> dict[str, float]:
    values = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if name not in JOINT_NAMES:
            sys.exit(f"Joint inconnu : {name!r} (attendu : {', '.join(JOINT_NAMES)})")
        values[name] = float(value)
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IK SO-ARM101 + commande de position (dry-run par défaut)")
    cmd = p.add_mutually_exclusive_group()
    cmd.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     help="point cible en mètres, repère base du bras (résolu par IK)")
    cmd.add_argument("--joints", nargs="+", metavar="JOINT=DEG",
                     help="angles absolus en degrés (0 = milieu de la plage), joints omis inchangés")
    cmd.add_argument("--sequence", type=Path, metavar="FICHIER", help="JSON : étapes exécutées dans l'ordre")
    cmd.add_argument("--status", action="store_true", help="lecture seule de la position courante")
    for angle, meaning in (("yaw", "vérifié seulement, imposé par la position"),
                           ("pitch", "0 = horizontal, -90 = vers le bas"),
                           ("roll", "autour de l'axe de la pince, = -wrist_roll")):
        p.add_argument(f"--{angle}", type=float, default=None, help=f"avec --point, en degrés ({meaning})")
    p.add_argument("--port", default=os.environ.get("ROBOT_PORT"), help="port série (défaut : $ROBOT_PORT)")
    p.add_argument("--calib", type=Path, default=Path(os.environ.get("CALIB_FILE", DEFAULT_CALIB_FILE)),
                   help="fichier de calibration Lerobot, utilisé sans port et pour comparaison")
    p.add_argument("--gripper", type=float, default=None, help="gripper en %% (0 = range_min, 100 = range_max)")
    p.add_argument("--settle", type=float, default=DEFAULT_SETTLE_S,
                   help=f"pause entre étapes d'une séquence, en s (défaut {DEFAULT_SETTLE_S})")
    p.add_argument("--move-time", type=float, default=DEFAULT_MOVE_S,
                   help=f"durée du mouvement vers chaque cible, en s (défaut {DEFAULT_MOVE_S})")
    p.add_argument("--min-z", type=float, default=DEFAULT_MIN_Z_M,
                   help=f"hauteur mini du gripper au-dessus de la table, en m (défaut {DEFAULT_MIN_Z_M})")
    p.add_argument("--tool", default=None,
                   help="outil de tools.json dont le point va en (x, y, z) (défaut : celui de la séquence, "
                        "sinon le 'default' de tools.json)")
    p.add_argument("--execute", action="store_true", help="écrire réellement sur le bus (implique --port)")
    return p


def print_ee(pose: dict[str, float], kin: ArmKinematics) -> None:
    x, y, z = kin.tool_transform(pose)[:3, 3]
    line = f"outil {kin.tool} : x={x:+.3f}  y={y:+.3f}  z={z:+.3f} m"
    if kin.supports_orientation:
        line += "  |  yaw={:+.1f}  pitch={:+.1f}  roll={:+.1f}°".format(*kin.orientation(pose))
    print(line)


def main() -> int:
    args = build_parser().parse_args()
    if not (args.point or args.joints or args.sequence or args.status):
        sys.exit("choisir --point, --joints, --sequence ou --status")
    if (args.execute or args.status) and not args.port:
        sys.exit("un port est requis (--port ou ROBOT_PORT)")
    orientation = {k: v for k, v in (("yaw", args.yaw), ("pitch", args.pitch), ("roll", args.roll))
                   if v is not None}
    if orientation and not args.point:
        sys.exit("--yaw/--pitch/--roll s'utilisent avec --point (dans une séquence : clé \"orientation\")")

    calib = resolve_calibration(args.port, args.calib)

    # Pose de départ : réelle si le bras est accessible, milieu de plage sinon.
    if args.port:
        with connect_arm(args.port) as bus:
            initial_raw = read_present_raw(bus)
    else:
        initial_raw, _ = pose_to_raw(neutral_pose(), calib)
    initial = pose_from_raw(initial_raw, calib)

    steps, sequence_tool = [], None
    if args.sequence:
        steps, sequence_tool = load_sequence(args.sequence)
    elif not args.status:
        step = {"point": args.point} if args.point else {"joints": parse_joints(args.joints)}
        if orientation:
            step["orientation"] = orientation
        if args.gripper is not None:
            step["gripper"] = args.gripper
        steps = [step]
    kin = ArmKinematics(args.tool or sequence_tool)

    if args.status:
        print(f"[status] position courante sur {args.port}")
        print_pose(initial_raw, calib)
        print_ee(initial, kin)
        return 0

    planned = plan_sequence(steps, calib, initial, kin)
    problems = check_plan(initial_raw, planned, calib, args.min_z, kin)

    print(f"\n[départ] (outil {kin.tool})")
    print_ee(initial, kin)
    for step in planned:
        print(f"\n[{step['label']}] {step['desc']}")
        print_pose(step["raw"], calib, step["clipped"])
        print_ee(pose_from_raw(step["raw"], calib), kin)
        error = "" if step["ik_error"] is None else f"erreur IK {step['ik_error'] * 1000:.1f} mm, "
        if step["wanted"]:
            error += f"{step['orientation_error']:.1f}°, "
        print(f"{error}gripper au plus bas sur le trajet : z={step['path_min_z'] * 1000:.0f} mm")

    if problems:
        print("\n[problèmes]\n  " + "\n  ".join(problems))
        if args.execute:
            sys.exit("[refusé] rien n'a été écrit sur le bus")
    if not args.execute:
        print("\n[dry-run] rien écrit -- ajouter --execute (avec --port) pour déplacer le bras")
        return 0

    with connect_arm(args.port) as bus:
        current = read_present_raw(bus)
        for step in planned:
            settle = step["settle_s"] if step["settle_s"] is not None else args.settle
            print(f"  -> {step['label']} : {step['desc']} (mouvement {args.move_time:.1f}s, pause {settle:.1f}s)")
            for raw in interpolate_raw(current, step["raw"], args.move_time):
                write_goal_raw(bus, raw)
                time.sleep(1 / RATE_HZ)
            current = step["raw"]
            time.sleep(settle)
    print(f"[ok] terminé sur {args.port} -- couple maintenu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
