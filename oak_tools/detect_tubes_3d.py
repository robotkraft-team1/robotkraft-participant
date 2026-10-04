#!/usr/bin/env python3
"""
Détection 3D temps réel des éprouvettes et du rack pour RobotKraft
- Modèle : YOLOv8 Nano fine-tuné (weights/best_ft_combined.pt / weights/best.pt - mAP@50: 93.1%)
- Caméra : OAK-D Lite (RGB + Profondeur alignée)
- Analyse colorimétrique : Échantillonnage automatique (Jaune, Cyan, Magenta, ou Transparent / Vide)
- Sortie : Position spatiale 3D (X, Y, Z) en millimètres pour chaque objet
"""

import argparse
import os
import sys
import time
import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Privilégier le nouveau meilleur modèle fine-tuné
_best_candidates = [
    os.path.join(SCRIPT_DIR, "weights", "best_ft_combined.pt"),
    os.path.join(SCRIPT_DIR, "weights", "best.pt"),
]
DEFAULT_WEIGHTS = next((p for p in _best_candidates if os.path.exists(p)), os.path.join(SCRIPT_DIR, "weights", "best.pt"))


def classify_liquid_color(bgr_crop):
    """
    Analyse la couleur du tube / liquide au centre du tube :
    - TRANSPARENT : tube à essai vide (faible saturation et chromaticité)
    - JAUNE, CYAN, MAGENTA : liquide coloré
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return "INCONNU", (128, 128, 128)

    # Chromaticité (écart max - min des canaux BGR)
    chroma = np.max(bgr_crop, axis=2).astype(float) - np.min(bgr_crop, axis=2).astype(float)
    med_chroma = float(np.median(chroma))

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]

    # Masque des pixels saturés et suffisamment lumineux
    mask_sat = (sat > 40) & (val > 40)
    sat_ratio = float(np.mean(mask_sat))
    med_sat = float(np.median(sat))

    # Tube à essai vide / transparent : peu de saturation et chromaticité quasi nulle
    if med_chroma < 30 or sat_ratio < 0.20 or med_sat < 35:
        return "TRANSPARENT", (220, 220, 220)  # BGR Gris très clair / blanc

    hues = hue[mask_sat]
    if len(hues) == 0:
        return "TRANSPARENT", (220, 220, 220)

    med_hue = float(np.median(hues))

    # Plages de teinte OpenCV (0 - 180)
    if 15 <= med_hue <= 55:
        return "JAUNE", (0, 230, 255)       # BGR Jaune vif
    elif 75 <= med_hue <= 135:
        return "CYAN", (255, 220, 0)        # BGR Cyan
    else:
        return "MAGENTA", (50, 50, 255)     # BGR Magenta / Rouge


def parse_args():
    parser = argparse.ArgumentParser(description="Détecteur 3D d'éprouvettes et rack par couleur de liquide")
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help=f"Chemin des poids YOLO entraînés (Défaut: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Seuil de confiance de détection (Défaut: 0.35)",
    )
    parser.add_argument(
        "--min-depth",
        type=int,
        default=300,
        help="Distance min valide en mm (Défaut: 300 mm)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1500,
        help="Distance max valide en mm (Défaut: 1500 mm)",
    )
    return parser.parse_args()


def build_pipeline(width=960, height=540, fps=30, min_depth=300, max_depth=1500):
    """Construit le pipeline DepthAI stéréo haute précision avec alignement RGB."""
    pipeline = dai.Pipeline()

    # Caméra RGB (CAM_A)
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 2)  # 1920x1080 -> 960x540
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(fps)

    # Monochromes stéréo (CAM_B & CAM_C)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(fps)
    mono_left.initialControl.setAutoExposureCompensation(-3)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(fps)
    mono_right.initialControl.setAutoExposureCompensation(-3)

    # Moteur Stéréo avec alignement pixel-à-pixel sur CAM_A
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    cfg = stereo.initialConfig.get()
    cfg.postProcessing.thresholdFilter.minRange = min_depth
    cfg.postProcessing.thresholdFilter.maxRange = max_depth
    stereo.initialConfig.set(cfg)

    # Liaisons
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline


def main():
    args = parse_args()
    width, height = 960, 540

    # 1. Résolution et chargement modèle YOLO
    weights_path = args.weights
    if not os.path.exists(weights_path):
        candidates = [
            os.path.join(SCRIPT_DIR, "weights", "best_ft_combined.pt"),
            os.path.join(SCRIPT_DIR, "weights", "best.pt"),
            os.path.join(SCRIPT_DIR, "..", "oak_tools", "weights", "best_ft_combined.pt"),
            os.path.join(SCRIPT_DIR, "..", "oak_tools", "weights", "best.pt"),
            "/home/plr/PycharmProjects/oak_tools/weights/best_ft_combined.pt",
            "/home/plr/PycharmProjects/oak_tools/weights/best.pt",
            os.path.join(SCRIPT_DIR, "..", "runs", "detect", "oak_tools", "runs", "tube_detector_ft_combined", "weights", "best.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = os.path.abspath(c)
                break

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Chargement du modèle YOLO (meilleur modèle fine-tuné) : {weights_path} ({device.upper()})...")
    try:
        model = YOLO(weights_path)
        model.to(device)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger les poids : {e}")
        sys.exit(1)

    # 2. Démarrage OAK-D Lite
    print("[INFO] Démarrage du flux OAK-D Lite...")
    pipeline = build_pipeline(width, height, fps=30, min_depth=args.min_depth, max_depth=args.max_depth)

    try:
        device_ctx = dai.Device(pipeline)
    except Exception as e:
        print(f"[ERREUR] Caméra OAK-D inaccessible : {e}")
        sys.exit(1)

    with device_ctx as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=4, blocking=False)

        # Focale sténopé
        calib = dev.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx_cam, cy_cam = intrinsics[0][2], intrinsics[1][2]

        win_name = "RobotKraft - Detection 3D Eprouvettes & Rack"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 65)
        print("  DETECTION 3D ACTIVE !")
        print("  - Tubes détectés : JAUNE / CYAN / MAGENTA / TRANSPARENT (vide)")
        print("  - Coordonnées 3D réelles : X, Y, Z en mm par rapport à la caméra")
        print("  - Touche 'c' : Imprimer la liste des objets détectés")
        print("  - Touche 'q' : Quitter")
        print("=" * 65 + "\n")

        fps_time = time.time()
        fps_counter = 0
        fps_display = 0.0

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            rgb_frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()

            # FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # Inférence YOLO
            results = model.predict(rgb_frame, conf=args.conf, verbose=False)[0]

            detected_items = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]  # 'tube' ou 'rack'
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Bornes
                x1_c, x2_c = max(0, x1), min(width, x2)
                y1_c, y2_c = max(0, y1), min(height, y2)

                # Profondeur dans la boîte
                roi_depth = depth_frame[y1_c:y2_c, x1_c:x2_c]
                valid_depth = roi_depth[(roi_depth >= args.min_depth) & (roi_depth <= args.max_depth)]

                if len(valid_depth) >= 10:
                    z_mm = float(np.median(valid_depth))
                    u_center = (x1 + x2) / 2.0
                    v_center = (y1 + y2) / 2.0
                    x_mm = (u_center - cx_cam) * z_mm / fx
                    y_mm = (v_center - cy_cam) * z_mm / fy

                    if cls_name == "tube":
                        # Échantillonner le liquide au centre du tube en évitant les bords de la table
                        bw = x2_c - x1_c
                        bh = y2_c - y1_c
                        if bh >= bw:
                            cw = max(1, int(bw * 0.15))
                            crop_liquid = rgb_frame[
                                max(0, int(y1_c + 0.25 * bh)) : min(height, int(y1_c + 0.85 * bh)),
                                max(0, x1_c + cw) : min(width, x2_c - cw),
                            ]
                        else:
                            ch = max(1, int(bh * 0.15))
                            crop_liquid = rgb_frame[
                                max(0, y1_c + ch) : min(height, y2_c - ch),
                                max(0, int(x1_c + 0.25 * bw)) : min(width, int(x1_c + 0.85 * bw)),
                            ]

                        if crop_liquid.size == 0:
                            crop_liquid = rgb_frame[y1_c:y2_c, x1_c:x2_c]

                        color_name, draw_color = classify_liquid_color(crop_liquid)
                        label = f"TUBE {color_name}"
                    else:
                        label = "RACK"
                        draw_color = (0, 255, 0)

                    detected_items.append({
                        "type": cls_name,
                        "label": label,
                        "conf": conf,
                        "x": x_mm,
                        "y": y_mm,
                        "z": z_mm,
                    })

                    # Rendu visuel
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), draw_color, 2)
                    cv2.circle(rgb_frame, (int(u_center), int(v_center)), 4, draw_color, -1)

                    txt_title = f"{label} ({conf*100:.0f}%)"
                    txt_coords = f"X:{int(x_mm):+4d} Y:{int(y_mm):+4d} Z:{int(z_mm)}mm"

                    (tw, _), _ = cv2.getTextSize(txt_title, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                    banner_w = max(210, tw + 12)
                    cv2.rectangle(rgb_frame, (x1, max(0, y1 - 36)), (x1 + banner_w, max(0, y1)), (0, 0, 0), -1)
                    cv2.putText(rgb_frame, txt_title, (x1 + 4, max(14, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, draw_color, 2)
                    cv2.putText(rgb_frame, txt_coords, (x1 + 4, max(30, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            # Bandeau HUD
            cv2.rectangle(rgb_frame, (10, 10), (width - 10, 42), (0, 0, 0), -1)
            hud = f"YOLOv8 Nano ({device}) | FPS: {fps_display:.1f} | Objets: {len(detected_items)} | 'c': print | 'q': quit"
            cv2.putText(rgb_frame, hud, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)

            cv2.imshow(win_name, rgb_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                print("\n--- OBJETS DETECTES SUR LA TABLE ---")
                if not detected_items:
                    print("  Aucun objet détecté.")
                for it in detected_items:
                    print(
                        f"  -> {it['label']:<18} conf={it['conf']:.2f} | "
                        f"X={it['x']:+6.1f} mm, Y={it['y']:+6.1f} mm, Z={it['z']:6.1f} mm"
                    )
                print("------------------------------------\n")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
