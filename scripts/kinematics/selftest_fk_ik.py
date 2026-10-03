"""Self-test hors matériel de l'IK de ik.py (placo + URDF SO-101), sans toucher au bras.

Depuis la pose de repos (bras replié), des points éloignés doivent être atteints à
moins de 1 mm, et un point hors de portée doit être détecté comme tel.
    .venv/bin/python scripts/kinematics/selftest_fk_ik.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ik  # noqa: E402

REST_DEG = {"shoulder_pan": 3.6, "shoulder_lift": -83.5, "elbow_flex": 95.9, "wrist_flex": 15.6, "wrist_roll": 97.4}
REACHABLE = [(0.20, 0.00, 0.20), (0.20, 0.12, 0.15), (0.20, -0.12, 0.15), (0.28, 0.00, 0.10)]
UNREACHABLE = (0.60, 0.00, 0.20)


def miss(target: tuple[float, float, float]) -> tuple[float, dict]:
    pose = ik.ik_point(*target, REST_DEG)
    return float(np.linalg.norm(ik.ee_transform(pose)[:3, 3] - target)), pose


def main() -> int:
    ok = True
    for target in REACHABLE:
        error, pose = miss(target)
        passed = error < 1e-3 and pose["wrist_roll"] == REST_DEG["wrist_roll"]
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {target} atteint à {error * 1000:.2f} mm, roll inchangé")
    error, _ = miss(UNREACHABLE)
    passed = error > ik.IK_TOLERANCE_M
    ok &= passed
    print(f"[{'PASS' if passed else 'FAIL'}] {UNREACHABLE} détecté hors de portée (manqué de {error * 1000:.0f} mm)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
