"""Bibliothèque de pilotage du bras SO-ARM101 : calibration, cinématique, bus, séquences.

C'est la partie importable de scripts/kinematics/ : l'interface en ligne de
commande est dans main.py, la visualisation 3D dans visualize.py.
Modules :
    calibration : ticks servos <-> degrés Lerobot, source de la calibration ;
    kinematics  : cinématique avant/inverse (placo), outils (tools.json), yaw/pitch/roll ;
    bus         : bus série Feetech (connexion, lecture, écriture, interpolation) ;
    sequence    : plan des étapes d'un JSON, vérifications de sécurité.

Exemple depuis un script qui calcule lui-même ses cibles (ex. vision) :

    from robot_arm import (ArmKinematics, resolve_calibration, connect_arm,
                           read_present_raw, write_goal_raw, interpolate_raw,
                           pose_from_raw, pose_to_raw, RATE_HZ)

    calib = resolve_calibration("/dev/ttyACM0", None)
    kin = ArmKinematics("centre_pince")
    with connect_arm("/dev/ttyACM0") as bus:
        current = read_present_raw(bus)
        pose = pose_from_raw(current, calib)
        pose.update(kin.ik_point(0.22, 0.05, 0.05, pose, pitch=-90))
        raw, _ = pose_to_raw(pose, calib)
        for step in interpolate_raw(current, raw, move_s=2.0):
            write_goal_raw(bus, step)
            time.sleep(1 / RATE_HZ)
"""

from .calibration import (
    ALL_MOTOR_NAMES,
    DEFAULT_CALIB_FILE,
    GRIPPER_DEFAULT,
    JOINT_NAMES,
    MAX_RES,
    calibration_mismatches,
    deg_limits,
    deg_to_raw,
    gripper_pct_to_raw,
    gripper_raw_to_pct,
    load_calibration_file,
    neutral_pose,
    pose_from_raw,
    pose_to_raw,
    raw_to_deg,
    read_servo_calibration,
    resolve_calibration,
)
from .kinematics import (
    DEFAULT_MIN_Z_M,
    GRIPPER_BOX,
    IK_CONVERGED_DEG,
    IK_CONVERGED_M,
    IK_MAX_ITERS,
    IK_TOLERANCE_DEG,
    IK_TOLERANCE_M,
    ORIENTATION_WEIGHT,
    PATH_SAMPLES,
    TOOLS_FILE,
    URDF_PATH,
    YAW_TOLERANCE_DEG,
    ArmKinematics,
    load_tools,
    rotation_angle_deg,
    rot_x,
    rot_y,
    rot_z,
    tool_matrix,
    urdf_with_tools,
    wrap_deg,
)
from .bus import (
    DEFAULT_MOVE_S,
    DEFAULT_SETTLE_S,
    RATE_HZ,
    connect_arm,
    interpolate_raw,
    read_present_raw,
    write_goal_raw,
)
from .sequence import check_plan, load_sequence, path_min_z, plan_sequence, target_rotation

__all__ = [
    # calibration
    "JOINT_NAMES", "ALL_MOTOR_NAMES", "MAX_RES", "GRIPPER_DEFAULT", "DEFAULT_CALIB_FILE",
    "load_calibration_file", "read_servo_calibration", "calibration_mismatches",
    "resolve_calibration", "raw_to_deg", "deg_to_raw", "deg_limits",
    "gripper_pct_to_raw", "gripper_raw_to_pct", "pose_from_raw", "pose_to_raw",
    "neutral_pose",
    # kinematics
    "URDF_PATH", "TOOLS_FILE", "IK_MAX_ITERS", "IK_CONVERGED_M", "IK_CONVERGED_DEG",
    "IK_TOLERANCE_M", "IK_TOLERANCE_DEG", "YAW_TOLERANCE_DEG", "ORIENTATION_WEIGHT",
    "DEFAULT_MIN_Z_M", "GRIPPER_BOX", "PATH_SAMPLES", "load_tools", "rot_x", "rot_y",
    "rot_z", "rotation_angle_deg", "wrap_deg", "tool_matrix", "urdf_with_tools",
    "ArmKinematics",
    # bus
    "DEFAULT_SETTLE_S", "DEFAULT_MOVE_S", "RATE_HZ", "connect_arm", "read_present_raw",
    "write_goal_raw", "interpolate_raw",
    # sequence
    "load_sequence", "plan_sequence", "target_rotation", "path_min_z", "check_plan",
]
