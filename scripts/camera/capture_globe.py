"""Constitue un dataset d'images du globe (pour annotation puis entraînement YOLO) avec les deux caméras :
OAK-D Lite (vue de scène, "top") et caméra poignet USB ("wrist"). Chaque capture enregistre une image par
caméra, prises au même instant, et la position des articulations du follower (utile plus tard pour relier
une pose du bras à un point du globe).

Fenêtre de preview : ESPACE/S = capture, A = capture auto on/off (toutes les AUTO_INTERVAL s), Q/Échap = quitter.

Variables d'environnement (voir make capture-globe) :
  OUT_DIR        dossier du dataset (défaut datasets/globe)
  CAMERAS        both | oak | wrist (défaut both)
  ARM            teleop : le follower suit le leader (on vise avec le leader)
                 read   : lecture seule, le follower n'est pas alimenté en couple et se bouge à la main
                 none   : pas de bras
  WRIST_DEVICE   défaut /dev/video2 ; TOP_WIDTH/TOP_HEIGHT (1920x1080), WRIST_WIDTH/WRIST_HEIGHT (1280x720)
  AUTO_INTERVAL  secondes entre deux captures auto (défaut 1.0)
  N_SHOTS        sans fenêtre : prend N captures toutes les AUTO_INTERVAL s puis quitte (test, ou sans écran)

Sortie (format attendu par YOLO / Label Studio / Roboflow : les labels .txt viendront à côté, même nom) :
  OUT_DIR/images/<cam>_<session>_<nnnn>.jpg
  OUT_DIR/captures.csv   une ligne par capture : fichiers, horodatage, articulations du follower
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from camera_utils import OAKCamera, V4L2Camera

OUT_DIR = Path(os.environ.get("OUT_DIR", "datasets/globe"))
CAMERAS = os.environ.get("CAMERAS", "both")
ARM = os.environ.get("ARM", "teleop")
AUTO_INTERVAL = float(os.environ.get("AUTO_INTERVAL", "1.0"))
N_SHOTS = int(os.environ.get("N_SHOTS", "0"))
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def chown(path):
    """Fichiers créés par root dans le conteneur : les rendre à l'utilisateur de l'hôte si HOST_UID est fourni"""
    uid, gid = os.environ.get("HOST_UID"), os.environ.get("HOST_GID")
    if uid and gid:
        try:
            os.chown(path, int(uid), int(gid))
        except OSError:
            pass


class Arm(threading.Thread):
    """Boucle bras à 30 Hz en arrière-plan : teleop leader -> follower, ou lecture seule du follower.
    Garde la dernière position lue ; une lecture ratée est ignorée au lieu d'arrêter la capture."""

    def __init__(self, mode):
        super().__init__(daemon=True)
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.mode = mode
        self.follower = SO101Follower(SO101FollowerConfig(port=os.environ["FOLLOWER_PORT"], id="follower_arm"))
        self.leader = None
        if mode == "teleop":
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

            self.leader = SO101Leader(SO101LeaderConfig(port=os.environ["LEADER_PORT"], id="leader_arm"))
        self.joints = {}
        self.errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def connect(self):
        if self.mode == "teleop":
            self.follower.connect()
            self.leader.connect()
        else:
            # Lecture seule : uniquement le bus (aucune écriture, pas de couple), calibration chargée depuis le JSON
            self.follower.bus.connect()

    def run(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                if self.leader is not None:
                    self.follower.send_action(self.leader.get_action())
                pos = self.follower.bus.sync_read("Present_Position")
                with self._lock:
                    self.joints = pos
            except Exception:
                self.errors += 1
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t0)))

    def latest(self):
        with self._lock:
            return dict(self.joints)

    def close(self):
        self._stop.set()
        self.join(timeout=1)
        if self.mode == "teleop":
            self.leader.disconnect()
            self.follower.disconnect()
        else:
            self.follower.bus.disconnect(disable_torque=False)


def open_cameras():
    cams = {}
    if CAMERAS in ("both", "oak"):
        top = OAKCamera(width=int(os.environ.get("TOP_WIDTH", 1920)), height=int(os.environ.get("TOP_HEIGHT", 1080)))
        cams["top"] = (top.open(), top.read_latest)
    if CAMERAS in ("both", "wrist"):
        wrist = V4L2Camera(
            device=os.environ.get("WRIST_DEVICE", "/dev/video2"),
            width=int(os.environ.get("WRIST_WIDTH", 1280)),
            height=int(os.environ.get("WRIST_HEIGHT", 720)),
        ).open()
        wrist.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # frame le plus récent, pas un frame en file d'attente
        cams["wrist"] = (wrist, wrist.read)
    if not cams:
        sys.exit(f"CAMERAS={CAMERAS} invalide (both, oak ou wrist)")
    for name, (cam, _) in cams.items():
        if not cam.isOpened():
            sys.exit(f"Caméra {name} inaccessible")
    return cams


def preview(frames, n, auto, arm):
    """Mosaïque des caméras redimensionnées à la même hauteur, avec l'état en surimpression"""
    h = 480
    tiles = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames.values()]
    img = np.hstack(tiles)
    status = f"{n} captures | auto {'ON' if auto else 'off'} | ESPACE=capture A=auto Q=quitter"
    if arm is not None and arm.errors:
        status += f" | lectures bras ratees: {arm.errors}"
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img


def main():
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for d in (OUT_DIR, img_dir):
        chown(d)
    csv_path = OUT_DIR / "captures.csv"
    new_csv = not csv_path.exists()
    session = datetime.now().strftime("%Y%m%d_%H%M%S")

    arm = None
    if ARM in ("teleop", "read"):
        arm = Arm(ARM)
        arm.connect()
        arm.start()
    elif ARM != "none":
        sys.exit(f"ARM={ARM} invalide (teleop, read ou none)")

    cams = open_cameras()
    print(f"Dataset : {OUT_DIR}  session {session}  caméras {list(cams)}  bras {ARM}", flush=True)

    n, auto, last_auto = 0, N_SHOTS > 0, 0.0
    headless = N_SHOTS > 0
    win = "Capture globe"
    if not headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if new_csv:
            writer.writerow(["capture", "timestamp", "session", "top", "wrist"] + JOINTS)
        try:
            while True:
                frames = {}
                for name, (_, read) in cams.items():
                    ok, frame = read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Lecture caméra {name} impossible")
                    frames[name] = frame

                key = -1
                if not headless:
                    cv2.imshow(win, preview(frames, n, auto, arm))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord("a"):
                        auto, last_auto = not auto, 0.0

                now = time.time()
                if key in (ord(" "), ord("s")) or (auto and now - last_auto >= AUTO_INTERVAL):
                    last_auto = now
                    capture_id = f"{session}_{n:04d}"
                    files = {}
                    for name, frame in frames.items():
                        path = img_dir / f"{name}_{capture_id}.jpg"
                        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        chown(path)
                        files[name] = path.name
                    joints = arm.latest() if arm is not None else {}
                    writer.writerow(
                        [capture_id, datetime.now().isoformat(timespec="milliseconds"), session,
                         files.get("top", ""), files.get("wrist", "")]
                        + [f"{joints[j]:.2f}" if j in joints else "" for j in JOINTS]
                    )
                    fcsv.flush()
                    n += 1
                    print(f"  capture {capture_id} : {', '.join(files.values())}", flush=True)
                    if headless and n >= N_SHOTS:
                        break
        finally:
            chown(csv_path)
            for cam, _ in cams.values():
                cam.release()
            if arm is not None:
                arm.close()
            cv2.destroyAllWindows()
    print(f"{n} captures enregistrées dans {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
