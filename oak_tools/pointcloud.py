#!/usr/bin/env python3

import argparse
import time
import cv2
import depthai as dai
import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(
        description="OAK-D 3D Point Cloud Viewer (Single Colored 3D View - Haute Precision)"
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=1.8,
        help="Distance maximale en mètres à afficher en 3D (Défaut : 1.8 m)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.35,
        help="Distance minimale en mètres à afficher en 3D (Défaut : 0.35 m, limite physique OAK-D-Lite)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="Pas d'échantillonnage de la grille (1 = 500k points pleine résolution, 2 = ~130k points fluide). Défaut : 2",
    )
    parser.add_argument(
        "--color-mode",
        type=str,
        choices=["rgb", "depth"],
        default="rgb",
        help="Coloration : 'rgb' (couleurs photo réelles) ou 'depth' (gradient thermique). Défaut : 'rgb'",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        choices=[1, 2],
        default=1,
        help="Facteur de décimation du moteur stéréo (1 = Haute Précision native, 2 = Ultra fluide 30 FPS). Défaut : 1",
    )
    parser.add_argument(
        "--spatial-filter",
        action="store_true",
        help="Activer le filtre spatial doux (comblement de micro-trous). Par défaut : désactivé pour éviter les bavures",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Pipeline stéréo calibrée haute précision alignée sur la caméra couleur
    pipeline = dai.Pipeline()

    width, height = 960, 540

    # Caméra couleur (CAM_A)
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 2)  # 960x540
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setFps(30)

    # Capteurs monochromes (CAM_B & CAM_C)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(30)
    # Réglages capteurs indispensables : compensation d'exposition (-3) pour supprimer le blanc brûlé
    # + netteté et contraste pour enrichir les micro-textures
    mono_left.initialControl.setAutoExposureCompensation(-3)
    mono_left.initialControl.setSharpness(2)
    mono_left.initialControl.setContrast(1)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(30)
    mono_right.initialControl.setAutoExposureCompensation(-3)
    mono_right.initialControl.setSharpness(2)
    mono_right.initialControl.setContrast(1)

    # Moteur stéréo avec Subpixel & Alignement RGB
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setSubpixel(True)
    # Extended Disparity DESACTIVE pour conserver actif le filtre médian 7x7 et éviter les points 3D aberrants
    stereo.setExtendedDisparity(False)
    stereo.setLeftRightCheck(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Filtre médian 7x7 matériel pleinement actif
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    # Post-traitement haute précision
    config = stereo.initialConfig.get()
    config.postProcessing.decimationFilter.decimationFactor = args.decimation
    config.algorithmControl.leftRightCheckThreshold = 5
    config.costMatching.confidenceThreshold = 230

    if args.spatial_filter:
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 1
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.spatialFilter.alpha = 0.35
        config.postProcessing.spatialFilter.delta = 8
    else:
        config.postProcessing.spatialFilter.enable = False

    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.temporalFilter.persistencyMode = (
        dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
    )
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 40

    config.postProcessing.thresholdFilter.minRange = int(args.min_dist * 1000)
    config.postProcessing.thresholdFilter.maxRange = int(args.max_dist * 1000)
    stereo.initialConfig.set(config)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Flux de sortie
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.video.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # 2. Démarrage et lecture des intrinsèques de calibration d'usine
    print("[INFO] Connexion a la camera OAK-D...")
    with dai.Device(pipeline) as device:
        print(f"[INFO] Connecte en USB : {device.getUsbSpeed().name}")

        calib = device.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, width, height)
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]
        print(f"[INFO] Intrinseques : fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        # Grille de coordonnées précalculée pour projection 3D rapide
        step = max(1, args.step)
        grid_u, grid_v = np.meshgrid(
            np.arange(0, width, step, dtype=np.float32),
            np.arange(0, height, step, dtype=np.float32),
        )
        x_norm = (grid_u - cx) / fx
        y_norm = (grid_v - cy) / fy
        grid_v_int = grid_v.astype(np.int32)
        grid_u_int = grid_u.astype(np.int32)

        # 3. Initialisation de la fenêtre 3D Open3D
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name="OAK-D : Nuage de points 3D Calibré",
            width=1280,
            height=800,
        )

        pcd = o3d.geometry.PointCloud()
        is_first_frame = True

        color_mode = {"mode": args.color_mode}

        def toggle_color_mode(v):
            color_mode["mode"] = "depth" if color_mode["mode"] == "rgb" else "rgb"
            print(f"[INFO] Mode couleur basculé sur : {color_mode['mode'].upper()}")
            return False

        # Touche 'c' pour alterner entre couleurs réelles RGB et heatmap de profondeur
        vis.register_key_callback(ord("C"), toggle_color_mode)
        vis.register_key_callback(ord("c"), toggle_color_mode)

        # Repère visuel d'axes 3D (X: Rouge, Y: Vert, Z: Bleu)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        vis.add_geometry(axes)

        # Réglage du rendu (points nets, fond sombre moderne)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.08, 0.08, 0.10])
        opt.point_size = 2.8

        print("\n" + "=" * 60)
        print("  CONTROLES 3D DU NUAGE DE POINTS :")
        print("  - Clic gauche + glisser : Rotation 3D de la vue")
        print("  - Roulette souris : Zoom avant / arrière")
        print("  - Molette cliquée (ou Maj + Clic) : Déplacement (Pan)")
        print("  - Touche 'c' : Alterner entre vraies couleurs RGB et Heatmap")
        print("  - Touche 'r' : Réinitialiser la vue")
        print("  - Touche 'q' : Quitter")
        print("=" * 60 + "\n")

        fps_counter = 0
        fps_start = time.time()

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            rgb_frame = in_rgb.getCvFrame()  # BGR
            depth_frame = in_depth.getFrame()  # mm (uint16)

            # Échantillonnage de la profondeur
            z_sample = depth_frame[grid_v_int, grid_u_int] * 0.001  # mètres

            # Filtrage selon la distance demandée
            valid_mask = (z_sample >= args.min_dist) & (z_sample <= args.max_dist)
            z_valid = z_sample[valid_mask]

            if len(z_valid) > 0:
                # Projection 3D (X, Y, Z réels en mètres)
                # Convention Open3D : X droite, Y haut, Z vers l'avant (profondeur)
                x_3d = x_norm[valid_mask] * z_valid
                y_3d = -y_norm[valid_mask] * z_valid  # Inverser Y pour que le haut soit en haut
                z_3d = z_valid

                points_3d = np.column_stack((x_3d, y_3d, z_3d))

                # Attribution des couleurs
                if color_mode["mode"] == "rgb":
                    # Vraies couleurs photographiques RGB de la caméra
                    bgr_sample = rgb_frame[grid_v_int, grid_u_int]
                    bgr_valid = bgr_sample[valid_mask]
                    colors = bgr_valid[:, [2, 1, 0]] / 255.0  # BGR vers RGB normalisé [0, 1]
                else:
                    # Heatmap de distance (Rouge = proche, Bleu = loin)
                    norm_z = np.clip(
                        (z_valid - args.min_dist) / (args.max_dist - args.min_dist),
                        0.0,
                        1.0,
                    )
                    depth_u8 = (255 * (1.0 - norm_z)).astype(np.uint8)
                    heat_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)[:, 0, :]
                    colors = heat_bgr[:, [2, 1, 0]] / 255.0

                pcd.points = o3d.utility.Vector3dVector(points_3d)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                if is_first_frame:
                    vis.add_geometry(pcd)
                    # Positionner la vue initiale de face
                    ctr = vis.get_view_control()
                    ctr.set_lookat([0.0, 0.0, 1.0])
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, 1.0, 0.0])
                    ctr.set_zoom(0.7)
                    is_first_frame = False
                else:
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            # Compteur de FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                print(f"[3D FPS: {fps_counter}] ({len(pcd.points):,} points 3D actifs)")
                fps_counter = 0
                fps_start = time.time()

            if not vis.poll_events():
                break

        vis.destroy_window()

    print("[INFO] Fermé proprement.")


if __name__ == "__main__":
    main()
