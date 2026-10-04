"""Outil interactif de calibration de la balance des blancs (offsets LAB a/b) via trackbars OpenCV"""

import json
import os
import sys

import cv2
import numpy as np

from camera_utils import V4L2Camera, ZMQCamera

CAMERA_CONFIG = "datasets/camera_config.json"

OFFSET = 50 # valeur trackbar centrale = 0 correction


def apply_wb(frame, a_off, b_off):
    """Applique une correction de balance des blancs en décalant les canaux a/b de l'espace LAB"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] += a_off
    lab[:, :, 2] += b_off
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def main():
    """Boucle caméra avec trackbars a/b en temps réel. S sauve les offsets"""
    # WB_DEVICE=/dev/videoX : webcam USB (caméra poignet) ; sinon flux ZMQ de l'OAK-D Lite
    device = os.environ.get("WB_DEVICE")
    cap = V4L2Camera(device) if device else ZMQCamera()
    cap.open()
    if not cap.isOpened():
        print(f"Caméra {device} inaccessible." if device else "OAK-D Lite inaccessible.")
        sys.exit(1)

    win = "Preview WB  |  Q=quitter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.createTrackbar("a_offset (-50..+50)", win, OFFSET, 100, lambda x: None)
    cv2.createTrackbar("b_offset (-50..+50)", win, OFFSET, 100, lambda x: None)

    print("Ajuster trackbars jusqu'à blanc = blanc. S pour sauver.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        a_off = cv2.getTrackbarPos("a_offset (-50..+50)", win) - OFFSET
        b_off = cv2.getTrackbarPos("b_offset (-50..+50)", win) - OFFSET

        corrected = apply_wb(frame, a_off, b_off)

        info = f"a={a_off:+d}  b={b_off:+d}  (S=sauver  Q=quitter)"
        cv2.putText(corrected, info, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.imshow(win, corrected)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            os.makedirs(os.path.dirname(CAMERA_CONFIG), exist_ok=True)
            cfg = {}
            if os.path.exists(CAMERA_CONFIG):
                with open(CAMERA_CONFIG) as f:
                    cfg = json.load(f)
            cfg["white_balance"] = {"a_offset": a_off, "b_offset": b_off}
            with open(CAMERA_CONFIG, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"Sauvé: a={a_off} b={b_off} -> {CAMERA_CONFIG}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
