"""Ligne de commande du bras SO-ARM101 : donner un point (ou des angles), le bras s'y rend.

La logique est dans le paquet robot_arm/ (importable par un script externe,
ex. la vision) ; ce fichier ne fait que l'interface --point/--joints/
--sequence/--status et l'exécution.

Commandes :
    --point X Y Z      point en mètres, repère base du bras, résolu par IK ;
    --joints J=DEG ... angles absolus en degrés (0 = milieu de la plage) ;
    --sequence FICHIER liste d'étapes exécutées dans l'ordre (voir sequence_points.json) ;
    --status           lecture seule de la position courante (angles + x, y, z de l'outil).

Outil et orientation : voir les docstrings de robot_arm.kinematics (tools.json,
yaw/pitch/roll).

Avant toute exécution, le plan est refusé si un point (avec son orientation)
est hors de portée, si le yaw demandé diffère de plus de 5° de celui imposé
par la position, si une cible est en butée, ou si le gripper ou l'outil passe
sous --min-z (cibles et trajets).

DRY-RUN par défaut. --execute écrit sur le bus ; chaque cible est rejointe par
interpolation linéaire en --move-time secondes (comme visualize.py), et le
couple reste actif au débranchement (disable_torque=False).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robot_arm import (  # noqa: E402
    JOINT_NAMES,
    RATE_HZ,
    DEFAULT_CALIB_FILE,
    DEFAULT_MIN_Z_M,
    DEFAULT_MOVE_S,
    DEFAULT_SETTLE_S,
    ArmKinematics,
    check_plan,
    connect_arm,
    deg_limits,
    gripper_raw_to_pct,
    interpolate_raw,
    load_sequence,
    neutral_pose,
    plan_sequence,
    pose_from_raw,
    pose_to_raw,
    raw_to_deg,
    read_present_raw,
    resolve_calibration,
    write_goal_raw,
)


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


def print_pose(raw: dict[str, int], calib: dict, clipped: list[str] = ()) -> None:
    print(f"{'Joint':<15} {'deg':>8} {'ticks':>6}   {'plage deg':>17}")
    for n in JOINT_NAMES:
        lo, hi = deg_limits(n, calib)
        flag = "  <- butée" if n in clipped else ""
        print(f"{n:<15} {raw_to_deg(n, raw[n], calib):>+8.1f} {raw[n]:>6}   [{lo:>+6.1f}, {hi:>+6.1f}]{flag}")
    print(f"{'gripper':<15} {gripper_raw_to_pct(raw['gripper'], calib):>7.0f}% {raw['gripper']:>6}")


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
