#!/usr/bin/env python3
"""
Script de collecte de dataset photo double caméra pour RobotKraft
Capture synchronisée ou individuelle :
- Caméra de scène : OAK-D Lite (1080p via depthai)
- Caméra de poignet robot : InnoMaker U20CAM (1080p via V4L2)

Utilisation :
  .venv/bin/python oak_tools/collect_dataset.py

Touches :
  [ESPACE] : Prendre une photo sur les DEUX caméras en même temps (1080p)
  [o]      : Prendre une photo sur la caméra OAK-D seule
  [w]      : Prendre une photo sur la caméra Poignet seule
  [q]/ESC  : Quitter
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import cv2
import depthai as dai
import numpy as np


def find_wrist_camera():
    """Tente de trouver automatiquement le bon device pour la caméra InnoMaker."""
    # 1. Vérifier si /dev/video2 existe
    if os.path.exists("/dev/video2"):
        return "/dev/video2"

    # 2. Chercher dans les liens by-id
    by_id = glob.glob("/dev/v4l/by-id/*InnoMaker*") + glob.glob("/dev/v4l/by-id/*U20CAM*")
    if by_id:
        return os.path.realpath(by_id[0])

    # 3. Fallback sur index 0
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Collecteur de photos multi-caméras pour dataset")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="dataset_raw",
        help="Dossier où sauvegarder les photos (Défaut: dataset_raw)",
    )
    parser.add_argument(
        "--wrist-device",
        type=str,
        default=None,
        help="Chemin du device V4L2 ou index pour la caméra poignet (ex: /dev/video2 ou 2)",
    )
    return parser.parse_args()


def init_oak_pipeline():
    """Crée le pipeline OAK-D en 1080p natif."""
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(30)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam_rgb.video.link(xout.input)
    return pipeline


def main():
    args = parse_args()

    # Dossiers de sortie
    out_dir = os.path.abspath(args.output_dir)
    dir_oak = os.path.join(out_dir, "oak")
    dir_wrist = os.path.join(out_dir, "wrist")
    dir_all = os.path.join(out_dir, "all")  # Regroupe tout pour import direct dans Roboflow
    for d in [dir_oak, dir_wrist, dir_all]:
        os.makedirs(d, exist_ok=True)

    # Compter les photos déjà existantes
    oak_count = len(glob.glob(os.path.join(dir_oak, "*.jpg")))
    wrist_count = len(glob.glob(os.path.join(dir_wrist, "*.jpg")))

    print("=" * 65)
    print("  COLLECTEUR DE PHOTOS DATASET ROBOTKRAFT")
    print(f"  Dossier de sauvegarde : {out_dir}")
    print(f"  Photos existantes     : OAK={oak_count}, Poignet={wrist_count}")
    print("=" * 65)

    # 1. Initialisation Caméra Poignet (InnoMaker)
    wrist_dev = args.wrist_device if args.wrist_device else find_wrist_camera()
    print(f"[INFO] Ouverture caméra poignet ({wrist_dev})...")
    cap_wrist = cv2.VideoCapture(int(wrist_dev) if str(wrist_dev).isdigit() else wrist_dev)

    wrist_available = False
    if cap_wrist.isOpened():
        cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, test_frame = cap_wrist.read()
        if ret and test_frame is not None:
            wrist_available = True
            print(f"[OK] Caméra poignet prête : {test_frame.shape[1]}x{test_frame.shape[0]}")
        else:
            print("[AVERTISSEMENT] Caméra poignet détectée mais impossible de lire un frame.")
    else:
        print("[AVERTISSEMENT] Caméra poignet inaccessible. Vérifiez le câble USB.")

    # 2. Initialisation Caméra OAK-D Lite
    print("[INFO] Connexion à la caméra OAK-D Lite...")
    oak_available = False
    oak_device = None
    q_oak = None

    try:
        pipeline = init_oak_pipeline()
        oak_device = dai.Device(pipeline)
        q_oak = oak_device.getOutputQueue("rgb", maxSize=4, blocking=False)
        # Test première lecture
        for _ in range(5):
            _ = q_oak.get()
        oak_available = True
        print("[OK] Caméra OAK-D Lite prête : 1920x1080")
    except Exception as e:
        print(f"[AVERTISSEMENT] OAK-D Lite indisponible : {e}")

    if not oak_available and not wrist_available:
        print("[ERREUR] Aucune des deux caméras n'a pu être initialisée. Abandon.")
        sys.exit(1)

    win_name = "RobotKraft - Capture Dataset Multi-Cameras"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

    # État du message de confirmation (flash visuel vert)
    last_saved_msg = ""
    saved_msg_time = 0.0

    preview_w, preview_h = 640, 360  # Taille de chaque panneau dans la fenêtre preview

    print("\n" + "-" * 65)
    print("  COMMANDES CLAVIER :")
    print("  [ESPACE] : Prendre une photo sur LES DEUX caméras")
    print("  [o]      : Photo OAK-D seule")
    print("  [w]      : Photo Poignet seule")
    print("  [q]/ESC  : Quitter")
    print("-" * 65 + "\n")

    try:
        while True:
            frame_oak = None
            frame_wrist = None

            # Lecture OAK
            if oak_available and q_oak is not None:
                in_oak = q_oak.tryGet()
                if in_oak is not None:
                    frame_oak = in_oak.getCvFrame()

            # Lecture Poignet
            if wrist_available and cap_wrist.isOpened():
                ret, frame_wrist = cap_wrist.read()
                if not ret:
                    frame_wrist = None

            # Panneau de gauche (OAK)
            if frame_oak is not None:
                p_oak = cv2.resize(frame_oak, (preview_w, preview_h))
            else:
                p_oak = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
                cv2.putText(p_oak, "OAK-D : En attente...", (40, preview_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(p_oak, f"OAK-D Lite (Total: {oak_count})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0) if oak_available else (100, 100, 100), 2)

            # Panneau de droite (Poignet)
            if frame_wrist is not None:
                p_wrist = cv2.resize(frame_wrist, (preview_w, preview_h))
            else:
                p_wrist = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
                cv2.putText(p_wrist, "Poignet : Indisponible", (40, preview_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(p_wrist, f"Poignet /dev/video2 (Total: {wrist_count})", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if wrist_available else (100, 100, 100), 2)

            # Assemblage côte-à-côte
            display = np.hstack([p_oak, p_wrist])

            # Bandeau HUD inférieur
            hud_h = 70
            hud = np.zeros((hud_h, display.shape[1], 3), dtype=np.uint8)

            now = time.time()
            if now - saved_msg_time < 1.5 and last_saved_msg:
                # Message flash vert lors d'une capture
                cv2.rectangle(hud, (0, 0), (display.shape[1], hud_h), (0, 120, 0), -1)
                cv2.putText(hud, f"[CAPTURE OK] {last_saved_msg}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                cv2.putText(
                    hud,
                    "[ESPACE] : Sauvegarder les 2 cameras  |  [o] : OAK seule  |  [w] : Poignet seule  |  [q] : Quitter",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    1,
                )
                cv2.putText(
                    hud,
                    f"Dossier : {out_dir}  |  Total photos dans all/ : {oak_count + wrist_count}",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                )

            final_view = np.vstack([display, hud])
            cv2.imshow(win_name, final_view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            # ------------------------------------------------------------------
            # Traitement des touches de capture
            # ------------------------------------------------------------------
            capture_oak = False
            capture_wrist = False

            if key == 32:  # ESPACE
                capture_oak = oak_available and (frame_oak is not None)
                capture_wrist = wrist_available and (frame_wrist is not None)
            elif key == ord("o"):
                capture_oak = oak_available and (frame_oak is not None)
            elif key == ord("w"):
                capture_wrist = wrist_available and (frame_wrist is not None)

            if capture_oak or capture_wrist:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                saved_types = []

                if capture_oak:
                    oak_count += 1
                    oak_filename = f"oak_{timestamp}_{oak_count:04d}.jpg"
                    oak_path = os.path.join(dir_oak, oak_filename)
                    oak_path_all = os.path.join(dir_all, oak_filename)

                    cv2.imwrite(oak_path, frame_oak, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(oak_path_all, frame_oak, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_types.append(f"OAK #{oak_count}")
                    print(f"  [+] Enregistré : {oak_filename} (1920x1080)")

                if capture_wrist:
                    wrist_count += 1
                    wrist_filename = f"wrist_{timestamp}_{wrist_count:04d}.jpg"
                    wrist_path = os.path.join(dir_wrist, wrist_filename)
                    wrist_path_all = os.path.join(dir_all, wrist_filename)

                    cv2.imwrite(wrist_path, frame_wrist, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(wrist_path_all, frame_wrist, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_types.append(f"Poignet #{wrist_count}")
                    print(f"  [+] Enregistré : {wrist_filename} (1920x1080)")

                last_saved_msg = " & ".join(saved_types)
                saved_msg_time = time.time()

    finally:
        if oak_device is not None:
            oak_device.close()
        if cap_wrist is not None and cap_wrist.isOpened():
            cap_wrist.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 65)
    print(f"  COLLECTE TERMINEE !")
    print(f"  Total photos OAK     : {oak_count}")
    print(f"  Total photos Poignet : {wrist_count}")
    print(f"  Toutes les photos sont prêtes dans : {dir_all}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
