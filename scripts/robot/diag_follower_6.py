#!/usr/bin/env python3
"""Stress du follower en lecture seule : N sync_read en lot sur les 6 servos, pour attraper
une comm instable (intermittente). Lecture brute (normalize=False). AUCUN mouvement,
bras non relache : disconnect(disable_torque=False)."""

import os
import time

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

PORT = os.environ.get("ROBOT_PORT", "/dev/ttyACM0")
N = int(os.environ.get("STRESS_N", "300"))
MOTOR_NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
motors = {MOTOR_NAMES[i]: Motor(id=i, model="sts3215", norm_mode=MotorNormMode.DEGREES)
          for i in MOTOR_NAMES}
keys = list(motors)

bus = FeetechMotorsBus(port=PORT, motors=motors)
bus.connect()
fails = 0
t0 = time.time()
try:
    for c in range(N):
        try:
            bus.sync_read("Present_Position", keys, normalize=False)
        except Exception as e:
            fails += 1
            if fails <= 5:
                print(f"  FAIL #{fails} (cycle {c+1}/{N}): {type(e).__name__}: {str(e)[:80]}")
finally:
    bus.disconnect(disable_torque=False)

dt = time.time() - t0
print(f"\n{N} sync_read en {dt:.2f}s -> {fails} echec(s) ({100*fails/N:.1f} %).")
print("disable_torque=False -> l'etat du bras est inchange.")
