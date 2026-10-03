"""Self-test hors matériel de l'IK de ik.py (placo + URDF SO-101), sans toucher au bras.

Depuis la pose de repos (bras replié), des points éloignés doivent être atteints à
moins de 1 mm, un point hors de portée doit être détecté, et un outil décalé et tourné
doit être placé par placo exactement là où tool_matrix le calcule.

Orientation : yaw/pitch/roll doivent reconstruire exactement la rotation de l'outil,
roll doit valoir -wrist_roll, yaw la direction de la pince vue du dessus ; des points
avec orientation doivent être atteints (< 1 mm, < 0.1°), et une orientation
impossible détectée.
    .venv/bin/python scripts/kinematics/selftest_fk_ik.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ik  # noqa: E402

REST_DEG = {"shoulder_pan": 3.6, "shoulder_lift": -83.5, "elbow_flex": 95.9, "wrist_flex": 15.6, "wrist_roll": 97.4}
REACHABLE = [(0.20, 0.00, 0.20), (0.20, 0.12, 0.15), (0.20, -0.12, 0.15), (0.28, 0.00, 0.10)]
UNREACHABLE = (0.60, 0.00, 0.20)
TEST_TOOL = {"xyz_m": [-0.01, 0.005, 0.03], "rpy_deg": [10.0, -20.0, 30.0]}
# (point, pitch, roll) atteignables avec l'outil centre_pince.
ORIENTED = [((0.22, 0.00, 0.04), -90.0, 90.0), ((0.24, 0.10, 0.12), -45.0, 0.0), ((0.28, 0.00, 0.16), 0.0, -30.0)]
# Pince verticale trop haut : wrist_flex en butée.
UNREACHABLE_ORIENTED = ((0.20, 0.00, 0.20), -90.0, 0.0)
# L'URDF arrondit pi/2 à 1.5708 : l'axe de la pince ne reste dans le plan du bras qu'à ~0.002° près.
CONVENTION_TOLERANCE_DEG = 0.01


def report(passed: bool, message: str) -> bool:
    print(f"[{'PASS' if passed else 'FAIL'}] {message}")
    return passed


def main() -> int:
    kin = ik.ArmKinematics("machoire_fixe")
    ok = True
    for target in REACHABLE:
        pose = kin.ik_point(*target, REST_DEG)
        error = np.linalg.norm(kin.tool_transform(pose)[:3, 3] - target)
        ok &= report(error < 1e-3 and pose["wrist_roll"] == REST_DEG["wrist_roll"],
                     f"{target} atteint à {error * 1000:.2f} mm, roll inchangé")
    pose = kin.ik_point(*UNREACHABLE, REST_DEG)
    error = np.linalg.norm(kin.tool_transform(pose)[:3, 3] - UNREACHABLE)
    ok &= report(error > ik.IK_TOLERANCE_M, f"{UNREACHABLE} détecté hors de portée (manqué de {error * 1000:.0f} mm)")

    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"default": "test", "tools": {"test": TEST_TOOL}}, f)
        f.flush()
        tool_kin = ik.ArmKinematics(tools_file=Path(f.name))
    expected = kin.tool_transform(REST_DEG) @ tool_kin.offset
    gap = np.abs(tool_kin.tool_transform(REST_DEG) - expected).max()
    ok &= report(gap < 1e-9, f"outil décalé et tourné : placo et tool_matrix concordent (écart {gap:.1e})")
    return 0 if check_orientation() and ok else 1


def check_orientation() -> bool:
    kin = ik.ArmKinematics("centre_pince")
    ok = True
    rng = np.random.default_rng(0)
    poses = [dict(zip(ik.JOINT_NAMES, rng.uniform(-90, 90, len(ik.JOINT_NAMES)))) for _ in range(20)]
    gap = max(ik.rotation_angle_deg(kin.rotation(*kin.orientation(p)), kin.tool_transform(p)[:3, :3]) for p in poses)
    ok &= report(gap < CONVENTION_TOLERANCE_DEG,
                 f"yaw/pitch/roll reconstruisent la rotation de l'outil (écart {gap:.1e}°)")

    turned = dict(REST_DEG, wrist_roll=REST_DEG["wrist_roll"] + 25.0)
    delta = ik.wrap_deg(kin.orientation(turned)[2] - kin.orientation(REST_DEG)[2])
    ok &= report(abs(delta + 25.0) < CONVENTION_TOLERANCE_DEG, f"wrist_roll +25° -> roll {delta:+.3f}°")

    for target, pitch, roll in ORIENTED:
        pose = kin.ik_point(*target, REST_DEG, pitch=pitch, roll=roll)
        tool = kin.tool_transform(pose)
        error = np.linalg.norm(tool[:3, 3] - target)
        angle_error = kin.orientation_error(pose, pitch, roll)
        ok &= report(error < 1e-3 and angle_error < 0.1,
                     f"{target} pitch={pitch:g} roll={roll:g} atteint à {error * 1000:.2f} mm, {angle_error:.3f}°")
        axis = tool[:3, 2]
        if abs(pitch) < 80:
            heading = np.degrees(np.arctan2(axis[1], axis[0]))
            yaw = kin.orientation(pose)[0]
            ok &= report(abs(ik.wrap_deg(heading - yaw)) < CONVENTION_TOLERANCE_DEG,
                         f"  yaw {yaw:+.1f}° = direction de la pince vue du dessus ({heading:+.1f}°)")

    target, pitch, roll = UNREACHABLE_ORIENTED
    pose = kin.ik_point(*target, REST_DEG, pitch=pitch, roll=roll)
    error = np.linalg.norm(kin.tool_transform(pose)[:3, 3] - target)
    ok &= report(error > ik.IK_TOLERANCE_M or kin.orientation_error(pose, pitch, roll) > ik.IK_TOLERANCE_DEG,
                 f"{target} pitch={pitch:g} détecté impossible (manqué de {error * 1000:.0f} mm)")
    return ok


if __name__ == "__main__":
    sys.exit(main())
