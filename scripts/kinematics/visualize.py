"""Visualisation 3D du bras SO-ARM101 dans le navigateur (meshcat).

Affiche l'URDF so101_new_calib avec les angles calculés comme Lerobot
(voir ik.py : degrés = (Present_Position - milieu de plage) * 360 / 4095).
L'URL exacte est affichée par meshcat au démarrage (ex. http://127.0.0.1:7000/static/).

    --watch               le modèle suit le bras réel (lecture seule, couple maintenu) ;
    --sequence FICHIER    rejoue en boucle la séquence au rythme de ik.py --execute (aucun mouvement) ;
    --point X Y Z         idem pour un seul point résolu par IK.

Usage :
    visualize.py --watch --port /dev/ttyACM0
    visualize.py --sequence scripts/kinematics/sequence_demo.json [--port /dev/ttyACM0]
    visualize.py --point 0.30 -0.10 0.25
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ik  # noqa: E402

STATUS_INTERVAL_S = 0.5


def load_model():
    import pinocchio as pin

    # assimp ne résout pas les chemins de meshes relatifs (assets/*.stl) de l'URDF.
    text = re.sub(r'filename="assets/', f'filename="{ik.URDF_PATH.parent / "assets"}/', ik.URDF_PATH.read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(text)
    try:
        model, _collision, visual = pin.buildModelsFromUrdf(f.name)
    finally:
        os.unlink(f.name)
    return model, visual


def make_viewer(model, visual):
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer

    viz = MeshcatVisualizer(model, None, visual, visual_data=pin.GeometryData(visual))
    viz.initViewer(open=False)
    viz.loadViewerModel()
    return viz


def config_from_raw(model, raw: dict[str, int], calib: dict) -> np.ndarray:
    """Ticks des 6 moteurs -> configuration pinocchio (radians, ordre de l'URDF)."""
    q = np.zeros(model.nq)
    for name in ik.JOINT_NAMES:
        q[model.joints[model.getJointId(name)].idx_q] = np.deg2rad(ik.raw_to_deg(name, raw[name], calib))
    # Lerobot ne relie pas le gripper à l'URDF : 0..100 % est étalé sur les limites du joint.
    idx = model.joints[model.getJointId("gripper")].idx_q
    frac = min(1.0, max(0.0, ik.gripper_raw_to_pct(raw["gripper"], calib) / 100))
    q[idx] = model.lowerPositionLimit[idx] + frac * (model.upperPositionLimit[idx] - model.lowerPositionLimit[idx])
    return q


def replay(viz, model, calib: dict, initial_raw: dict[str, int], planned: list[dict],
           move_s: float, settle_s: float) -> None:
    """Rejoue en boucle les consignes qu'enverrait ik.py --execute, au même rythme."""
    try:
        while True:
            current = initial_raw
            viz.display(config_from_raw(model, current, calib))
            time.sleep(settle_s)
            for step in planned:
                for raw in ik.interpolate_raw(current, step["raw"], move_s):
                    viz.display(config_from_raw(model, raw, calib))
                    time.sleep(1 / ik.RATE_HZ)
                current = step["raw"]
                time.sleep(step["settle_s"] if step["settle_s"] is not None else settle_s)
    except KeyboardInterrupt:
        print("\n[ok] visualisation arrêtée")


def watch(viz, model, port: str, calib: dict, interval_s: float) -> None:
    print(f"[watch] lecture toutes les {interval_s:.2f}s, Ctrl-C pour quitter")
    last_status = 0.0
    with ik.connect_arm(port) as bus:
        try:
            while True:
                raw = ik.read_present_raw(bus)
                viz.display(config_from_raw(model, raw, calib))
                now = time.time()
                if now - last_status >= STATUS_INTERVAL_S:
                    last_status = now
                    joints = " | ".join(f"{n}:{ik.raw_to_deg(n, raw[n], calib):+6.1f}" for n in ik.JOINT_NAMES)
                    print(f"{joints} | gripper:{ik.gripper_raw_to_pct(raw['gripper'], calib):4.0f}%", flush=True)
                time.sleep(interval_s)
        except KeyboardInterrupt:
            print("\n[ok] visualisation arrêtée")


def main() -> int:
    p = argparse.ArgumentParser(description="Visualisation 3D du bras (meshcat)")
    cmd = p.add_mutually_exclusive_group(required=True)
    cmd.add_argument("--watch", action="store_true", help="suivre le bras réel (lecture seule, implique --port)")
    cmd.add_argument("--sequence", type=Path, metavar="FICHIER", help="rejouer les cibles d'une séquence")
    cmd.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"), help="pose résolue par IK")
    p.add_argument("--port", default=os.environ.get("ROBOT_PORT"), help="port série (défaut : $ROBOT_PORT)")
    p.add_argument("--calib", type=Path, default=Path(os.environ.get("CALIB_FILE", ik.DEFAULT_CALIB_FILE)),
                   help="fichier de calibration Lerobot, utilisé sans port")
    p.add_argument("--interval", type=float, default=0.1, help="période de lecture en --watch, en s (défaut 0.1)")
    p.add_argument("--move-time", type=float, default=ik.DEFAULT_MOVE_S,
                   help=f"durée du mouvement vers chaque cible, en s (défaut {ik.DEFAULT_MOVE_S})")
    p.add_argument("--settle", type=float, default=ik.DEFAULT_SETTLE_S,
                   help=f"pause par défaut sur chaque cible, en s (défaut {ik.DEFAULT_SETTLE_S})")
    args = p.parse_args()
    if args.watch and not args.port:
        sys.exit("--watch implique --port (ou ROBOT_PORT)")

    calib = ik.resolve_calibration(args.port, args.calib)
    model, visual = load_model()
    viz = make_viewer(model, visual)

    if args.watch:
        watch(viz, model, args.port, calib, args.interval)
        return 0

    if args.port:
        with ik.connect_arm(args.port) as bus:
            initial_raw = ik.read_present_raw(bus)
    else:
        initial_raw, _ = ik.pose_to_raw(ik.neutral_pose(), calib)
    steps = ik.load_sequence(args.sequence) if args.sequence else [{"point": args.point}]
    planned = ik.plan_sequence(steps, calib, ik.pose_from_raw(initial_raw, calib))
    problems = ik.check_plan(initial_raw, planned, calib, ik.DEFAULT_MIN_Z_M)
    if problems:
        print("[problèmes] ik.py --execute refusera ce plan :\n  " + "\n  ".join(problems))
    replay(viz, model, calib, initial_raw, planned, args.move_time, args.settle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
