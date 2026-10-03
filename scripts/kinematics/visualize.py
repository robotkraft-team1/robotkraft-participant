"""Visualisation 3D du bras SO-ARM101 dans le navigateur (meshcat).

Affiche l'URDF so101_new_calib avec les angles calculés comme Lerobot
(voir ik.py : degrés = (Present_Position - milieu de plage) * 360 / 4095).
L'URL exacte est affichée par meshcat au démarrage (ex. http://127.0.0.1:7000/static/).

Les repères d'outil de tools.json sont dessinés en trièdres (x rouge, y vert, z bleu),
le grand étant l'outil sélectionné ; ils sont recalculés à chaque enregistrement de
tools.json, ce qui permet de régler un outil en regardant son repère bouger.
Les cibles d'une séquence (ou de --point) sont des sphères orange, avec un trièdre
fixe quand une orientation est demandée : le grand trièdre de l'outil doit s'y superposer.

    --frames              bras immobile (pose réelle avec --port, sinon pose neutre) ;
    --watch               le modèle suit le bras réel (lecture seule, couple maintenu) ;
    --sequence FICHIER    rejoue en boucle la séquence au rythme de ik.py --execute (aucun mouvement) ;
    --point X Y Z         idem pour un seul point résolu par IK (orientation : --yaw/--pitch/--roll).

Usage :
    visualize.py --frames [--port /dev/ttyACM0] [--tool centre_pince]
    visualize.py --watch --port /dev/ttyACM0
    visualize.py --sequence scripts/kinematics/sequence_orientation.json [--port /dev/ttyACM0]
    visualize.py --point 0.22 0.05 0.05 --pitch -90 --roll 45 [--tool centre_pince]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ik  # noqa: E402

STATUS_INTERVAL_S = 0.5
SELECTED_TRIAD_M = 0.06
OTHER_TRIAD_M = 0.03
TARGET_TRIAD_M = 0.04
TARGET_RADIUS_M = 0.006
TARGET_COLOR = 0xFF8800


def load_model():
    import pinocchio as pin

    with ik.urdf_with_tools({}) as urdf:
        model, _collision, visual = pin.buildModelsFromUrdf(str(urdf))
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


class ToolFrames:
    """Trièdres des outils de tools.json, accrochés à gripper_frame_link et rechargés si le fichier change."""

    def __init__(self, viz, model, selected: str, path: Path = ik.TOOLS_FILE):
        import pinocchio as pin

        self.viz, self.model, self.selected, self.path = viz, model, selected, path
        self.data = pin.Data(model)
        self.frame_id = model.getFrameId("gripper_frame_link")
        self.offsets: dict[str, np.ndarray] = {}
        self.mtime = None

    def _reload(self) -> None:
        import meshcat.geometry as g

        try:
            tools, _ = ik.load_tools(self.path)
        except (ValueError, KeyError) as e:
            print(f"[outils] {self.path.name} illisible, repères inchangés : {e}", flush=True)
            return
        self.viz.viewer["outils"].delete()
        self.offsets = {name: ik.tool_matrix(t) for name, t in tools.items()}
        for name in self.offsets:
            size = SELECTED_TRIAD_M if name == self.selected else OTHER_TRIAD_M
            self.viz.viewer["outils"][name].set_object(g.triad(size))
        print(f"[outils] {', '.join(self.offsets)} -- grand trièdre : {self.selected}", flush=True)

    def display(self, q: np.ndarray) -> None:
        """Affiche le bras dans la configuration q et place les trièdres des outils."""
        import pinocchio as pin

        self.viz.display(q)
        try:
            mtime = self.path.stat().st_mtime
        except OSError:  # fichier en cours d'enregistrement
            mtime = self.mtime
        if mtime != self.mtime:
            self.mtime = mtime
            self._reload()
        pin.framesForwardKinematics(self.model, self.data, q)
        gripper = self.data.oMf[self.frame_id].homogeneous
        for name, offset in self.offsets.items():
            self.viz.viewer["outils"][name].set_transform(gripper @ offset)


def show_targets(viz, planned: list[dict]) -> None:
    """Une sphère par point demandé, avec un trièdre si une orientation est demandée."""
    import meshcat.geometry as g

    viz.viewer["cibles"].delete()
    material = g.MeshLambertMaterial(color=TARGET_COLOR, opacity=0.6, transparent=True)
    for i, step in enumerate(planned):
        if step["target"] is None:
            continue
        node = viz.viewer["cibles"][f"{i + 1}"]
        pose = np.eye(4)
        pose[:3, 3] = step["target"]
        node["point"].set_object(g.Sphere(TARGET_RADIUS_M), material)
        if step["target_rotation"] is not None:
            pose[:3, :3] = step["target_rotation"]
            node["orientation"].set_object(g.triad(TARGET_TRIAD_M))
        node.set_transform(pose)


def hold(frames: ToolFrames, q: np.ndarray, interval_s: float) -> None:
    print("[frames] bras immobile ; modifier tools.json pour déplacer les repères (Ctrl-C pour quitter)")
    try:
        while True:
            frames.display(q)
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\n[ok] visualisation arrêtée")


def replay(frames: ToolFrames, calib: dict, initial_raw: dict[str, int], planned: list[dict],
           move_s: float, settle_s: float) -> None:
    """Rejoue en boucle les consignes qu'enverrait ik.py --execute, au même rythme."""

    def pause(seconds: float, raw: dict[str, int]) -> None:
        for _ in range(max(1, round(seconds * ik.RATE_HZ))):
            frames.display(config_from_raw(frames.model, raw, calib))
            time.sleep(1 / ik.RATE_HZ)

    try:
        while True:
            current = initial_raw
            pause(settle_s, current)
            for step in planned:
                for raw in ik.interpolate_raw(current, step["raw"], move_s):
                    frames.display(config_from_raw(frames.model, raw, calib))
                    time.sleep(1 / ik.RATE_HZ)
                current = step["raw"]
                pause(step["settle_s"] if step["settle_s"] is not None else settle_s, current)
    except KeyboardInterrupt:
        print("\n[ok] visualisation arrêtée")


def watch(frames: ToolFrames, port: str, calib: dict, interval_s: float) -> None:
    print(f"[watch] lecture toutes les {interval_s:.2f}s, Ctrl-C pour quitter")
    last_status = 0.0
    with ik.connect_arm(port) as bus:
        try:
            while True:
                raw = ik.read_present_raw(bus)
                frames.display(config_from_raw(frames.model, raw, calib))
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
    cmd.add_argument("--frames", action="store_true", help="bras immobile, pour régler les repères d'outil")
    cmd.add_argument("--watch", action="store_true", help="suivre le bras réel (lecture seule, implique --port)")
    cmd.add_argument("--sequence", type=Path, metavar="FICHIER", help="rejouer les cibles d'une séquence")
    cmd.add_argument("--point", nargs=3, type=float, metavar=("X", "Y", "Z"), help="pose résolue par IK")
    for angle in ("yaw", "pitch", "roll"):
        p.add_argument(f"--{angle}", type=float, default=None, help="avec --point, en degrés (voir ik.py)")
    p.add_argument("--port", default=os.environ.get("ROBOT_PORT"), help="port série (défaut : $ROBOT_PORT)")
    p.add_argument("--calib", type=Path, default=Path(os.environ.get("CALIB_FILE", ik.DEFAULT_CALIB_FILE)),
                   help="fichier de calibration Lerobot, utilisé sans port")
    p.add_argument("--tool", default=None, help="outil de tools.json mis en avant (et visé par l'IK)")
    p.add_argument("--interval", type=float, default=0.1, help="période de lecture en --watch, en s (défaut 0.1)")
    p.add_argument("--move-time", type=float, default=ik.DEFAULT_MOVE_S,
                   help=f"durée du mouvement vers chaque cible, en s (défaut {ik.DEFAULT_MOVE_S})")
    p.add_argument("--settle", type=float, default=ik.DEFAULT_SETTLE_S,
                   help=f"pause par défaut sur chaque cible, en s (défaut {ik.DEFAULT_SETTLE_S})")
    args = p.parse_args()
    if args.watch and not args.port:
        sys.exit("--watch implique --port (ou ROBOT_PORT)")
    orientation = {k: v for k, v in (("yaw", args.yaw), ("pitch", args.pitch), ("roll", args.roll))
                   if v is not None}
    if orientation and not args.point:
        sys.exit("--yaw/--pitch/--roll s'utilisent avec --point (dans une séquence : clé \"orientation\")")

    calib = ik.resolve_calibration(args.port, args.calib)
    model, visual = load_model()
    viz = make_viewer(model, visual)

    if args.watch:
        watch(ToolFrames(viz, model, args.tool or ik.load_tools()[1]), args.port, calib, args.interval)
        return 0

    if args.port:
        with ik.connect_arm(args.port) as bus:
            initial_raw = ik.read_present_raw(bus)
    else:
        initial_raw, _ = ik.pose_to_raw(ik.neutral_pose(), calib)

    if args.frames:
        hold(ToolFrames(viz, model, args.tool or ik.load_tools()[1]), config_from_raw(model, initial_raw, calib),
             args.interval)
        return 0

    if args.sequence:
        steps, sequence_tool = ik.load_sequence(args.sequence)
    else:
        steps, sequence_tool = [{"point": args.point, "orientation": orientation}], None
    kin = ik.ArmKinematics(args.tool or sequence_tool)
    planned = ik.plan_sequence(steps, calib, ik.pose_from_raw(initial_raw, calib), kin)
    problems = ik.check_plan(initial_raw, planned, calib, ik.DEFAULT_MIN_Z_M, kin)
    if problems:
        print("[problèmes] ik.py --execute refusera ce plan :\n  " + "\n  ".join(problems))
    show_targets(viz, planned)
    replay(ToolFrames(viz, model, kin.tool), calib, initial_raw, planned, args.move_time, args.settle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
