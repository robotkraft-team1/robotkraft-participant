"""Cinématique avant et inverse du SO-ARM101 (placo + URDF so101_new_calib).

Repère base (celui de l'URDF) : origine sous le socle, sur la table ; x vers
l'avant du bras quand shoulder_pan est au milieu de sa plage, y vers sa
gauche, z vers le haut.

Outils : le point amené en (x, y, z) est celui de l'outil choisi dans
tools.json : un repère défini par son décalage depuis gripper_frame_link
(face intérieure de la mâchoire fixe, 6 mm avant son bout).

Orientation (optionnelle, en degrés), donnée pour l'axe de la pince (l'axe de
wrist_roll, qui sort de la pince) :
    pitch : inclinaison de cet axe, 0 = horizontal, -90 = pince vers le bas ;
    yaw   : direction où il pointe vu du dessus, 0 = +x, 90 = +y. Le bras n'a
            que 5 axes : le yaw est imposé par la position (il vaut
            -shoulder_pan). Il n'est donc pas commandé, seulement vérifié s'il
            est donné ;
    roll  : rotation autour de cet axe, positive dans le sens horaire vu
            depuis le poignet (roll = -wrist_roll) ; 0 = wrist_roll au milieu
            de sa plage.
Sans orientation, l'IK contraint la position seule et wrist_roll reste
inchangé ; un angle omis reste libre (pitch) ou inchangé (roll).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .calibration import JOINT_NAMES

URDF_PATH = Path(__file__).resolve().parents[3] / "urdf" / "so101" / "so101_new_calib.urdf"
TOOLS_FILE = Path(__file__).resolve().parents[1] / "tools.json"

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
        """Orientation de l'outil (degrés, convention en tête de module) -> matrice 3x3 dans le repère base."""
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
