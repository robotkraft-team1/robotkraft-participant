"""Planification d'une séquence : étapes JSON -> ticks cibles, avec vérifications de sécurité.

Étape : {"return": "initial"} |
        {"point": [x, y, z], "orientation": {"yaw": deg, "pitch": deg, "roll": deg}} (chacun optionnel) |
        {"joints": {nom: deg}, "relative": bool}, avec "gripper" (%) optionnel.

Avant toute exécution, le plan est refusé si un point (avec son orientation) est
hors de portée, si le yaw demandé diffère de plus de YAW_TOLERANCE_DEG de celui
imposé par la position, si une cible est en butée, ou si le gripper ou l'outil
passe sous DEFAULT_MIN_Z_M (cibles et trajets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .calibration import JOINT_NAMES, pose_from_raw, pose_to_raw
from .kinematics import (
    IK_TOLERANCE_DEG,
    IK_TOLERANCE_M,
    PATH_SAMPLES,
    YAW_TOLERANCE_DEG,
    ArmKinematics,
    wrap_deg,
)


def load_sequence(path: Path) -> tuple[list[dict], str | None]:
    """Étapes de la séquence et outil qu'elle déclare ("tool", optionnel)."""
    with open(path) as f:
        data = json.load(f)
    steps = data["steps"] if isinstance(data, dict) else data
    if not steps:
        sys.exit(f"Séquence vide : {path}")
    return steps, data.get("tool") if isinstance(data, dict) else None


def plan_sequence(steps: list[dict], calib: dict, initial: dict[str, float], kin: ArmKinematics) -> list[dict]:
    """Ticks cibles de chaque étape, la pose courante étant portée d'une étape à l'autre."""
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
