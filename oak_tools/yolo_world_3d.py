#!/usr/bin/env python3
"""
OAK-D Lite + YOLO-World 3D Spatial Detection
Détection d'objets en vocabulaire ouvert (Zero-Shot) avec coordonnées 3D réelles (X, Y, Z) en millimètres.
Conçu pour le challenge RobotKraft (éprouvettes, racks, lego duplo, etc.).
"""

import argparse
import sys
import time
import cv2
import depthai as dai
import numpy as np
import torch
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser(
        description="OAK-D Lite + YOLO-World Détection 3D en Temps Réel"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-world.pt",
        help="Modèle YOLO-World ('yolov8s-world.pt', 'yolov8m-world.pt', 'yolov8x-world.pt'). Défaut: yolov8s-world.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Seuil de confiance de détection (0.05 à 0.90). Défaut: 0.20",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "test tube",
            "laboratory vial",
            "Eppendorf tube"
        ],
        help="Classes à détecter (séparées par un espace). Défaut: éprouvettes, rack, lego",
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
    """Construit le pipeline DepthAI aligné RGB + Stéréo haute précision."""
    pipeline = dai.Pipeline()

    # 1. Caméra RGB (CAM_A)
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 2)  # 1920x1080 -> 960x540
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(fps)

    # 2. Capteurs monochromes stéréo (CAM_B & CAM_C)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(fps)
    mono_left.initialControl.setAutoExposureCompensation(-3)
    mono_left.initialControl.setSharpness(2)
    mono_left.initialControl.setContrast(1)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(fps)
    mono_right.initialControl.setAutoExposureCompensation(-3)
    mono_right.initialControl.setSharpness(2)
    mono_right.initialControl.setContrast(1)

    # 3. Moteur Stéréo avec alignement pixel-à-pixel sur CAM_A
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

    # ==========================================================================
    # Initialisation YOLO-World
    # ==========================================================================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Accélération matérielle : {device.upper()}")
    if device.startswith("cuda"):
        print(f"[INFO] GPU détecté : {torch.cuda.get_device_name(0)}")

    print(f"[INFO] Chargement du modèle {args.model}...")
    try:
        model = YOLOWorld(args.model)
        model.to(device)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger YOLO-World : {e}")
        sys.exit(1)

    print(f"[INFO] Configuration des classes Zero-Shot :")
    for idx, c in enumerate(args.classes):
        print(f"   [{idx}] {c}")
    model.set_classes(args.classes)

    # Palette de couleurs distinctes par classe
    colors = [
        (255, 100, 0),   # Bleu-Cyan
        (0, 220, 255),   # Jaune
        (0, 50, 255),    # Rouge
        (0, 255, 120),   # Vert
        (255, 0, 200),   # Magenta
        (180, 180, 255), # Mauve
        (255, 255, 255), # Blanc
    ]

    # ==========================================================================
    # Initialisation OAK-D Lite
    # ==========================================================================
    print(f"[INFO] Démarrage du pipeline OAK-D Lite...")
    pipeline = build_pipeline(
        width=width,
        height=height,
        fps=30,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )

    try:
        device_ctx = dai.Device(pipeline)
    except Exception as e:
        print(f"[ERREUR] Impossible de se connecter à la caméra OAK-D : {e}")
        print("Vérifiez le branchement USB3 et les règles udev.")
        sys.exit(1)

    with device_ctx as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=4, blocking=False)

        # Récupération de la matrice intrinsèque pour la rétroprojection sténopé
        calib = dev.readCalibration()
        intrinsics = calib.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, width, height
        )
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx_cam, cy_cam = intrinsics[0][2], intrinsics[1][2]
        print(f"[INFO] Calibration optique : fx={fx:.1f}, fy={fy:.1f}, cx={cx_cam:.1f}, cy={cy_cam:.1f}")

        win_name = "RobotKraft - OAK-D Lite + YOLO-World 3D"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

        print("\n" + "=" * 65)
        print("  SYSTEME PRET !")
        print("  - Boîte verte/couleur : objet détecté avec position 3D (X, Y, Z)")
        print("  - Touche 'c' : afficher les coordonnées de tous les objets détectés")
        print("  - Touche 'q' : quitter")
        print("=" * 65 + "\n")

        fps_time = time.time()
        fps_counter = 0
        fps_display = 0.0

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            rgb_frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()  # profondeur brute en millimètres

            # Mesure FPS
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_display = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            # ------------------------------------------------------------------
            # Inférence YOLO-World
            # ------------------------------------------------------------------
            results = model.predict(rgb_frame, conf=args.conf, verbose=False)[0]

            detected_objects = []

            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Bornes de sécurité
                    x1_c, x2_c = max(0, x1), min(width, x2)
                    y1_c, y2_c = max(0, y1), min(height, y2)

                    # 1. Extraction de la profondeur dans la boîte (ROI)
                    roi = depth_frame[y1_c:y2_c, x1_c:x2_c]
                    valid = roi[(roi >= args.min_depth) & (roi <= args.max_depth)]

                    color = colors[cls_id % len(colors)]

                    if len(valid) >= 10:
                        # Profondeur médiane filtrée
                        z_mm = float(np.median(valid))

                        # 2. Rétroprojection sténopé vers repère caméra métrique (X, Y, Z)
                        u_center = (x1 + x2) / 2.0
                        v_center = (y1 + y2) / 2.0
                        x_mm = (u_center - cx_cam) * z_mm / fx
                        y_mm = (v_center - cy_cam) * z_mm / fy

                        detected_objects.append({
                            "name": cls_name,
                            "conf": conf,
                            "x": x_mm,
                            "y": y_mm,
                            "z": z_mm,
                            "box": (x1, y1, x2, y2),
                        })

                        txt_main = f"{cls_name} ({conf*100:.0f}%)"
                        txt_3d = f"X:{int(x_mm):+4d} Y:{int(y_mm):+4d} Z:{int(z_mm)}mm"

                        # Boîte et réticule central
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.circle(rgb_frame, (int(u_center), int(v_center)), 4, color, -1)

                        # Étiquettes
                        cv2.rectangle(rgb_frame, (x1, max(0, y1 - 38)), (x1 + 220, max(0, y1)), (0, 0, 0), -1)
                        cv2.putText(rgb_frame, txt_main, (x1 + 4, max(14, y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                        cv2.putText(rgb_frame, txt_3d, (x1 + 4, max(30, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    else:
                        # Détecté mais profondeur indisponible (trop proche ou caché)
                        cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                        cv2.putText(
                            rgb_frame,
                            f"{cls_name} (Z: indisponible)",
                            (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (120, 120, 120),
                            1,
                        )

            # ------------------------------------------------------------------
            # Bandeau HUD supérieur
            # ------------------------------------------------------------------
            cv2.rectangle(rgb_frame, (10, 10), ( width - 10, 45), (0, 0, 0), -1)
            hud = f"YOLO-World ({device}) | FPS: {fps_display:.1f} | Objets: {len(detected_objects)} | 'c': print | 'q': quit"
            cv2.putText(rgb_frame, hud, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

            cv2.imshow(win_name, rgb_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                print("\n--- OBJETS DETECTES ACTUELLEMENT ---")
                if not detected_objects:
                    print("  Aucun objet valide détecté.")
                for obj in detected_objects:
                    print(
                        f"  -> {obj['name']:<20} conf={obj['conf']:.2f} | "
                        f"X={obj['x']:+6.1f} mm, Y={obj['y']:+6.1f} mm, Z={obj['z']:6.1f} mm"
                    )
                print("------------------------------------\n")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
