#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
import time

# ==============================================================================
# 1. Pipeline stéréo haute précision OAK-D-Lite avec alignement RGB
# ==============================================================================
pipeline = dai.Pipeline()

# Caméra couleur (CAM_A) - 960x540
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setIspScale(1, 2)  # 960x540
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setFps(30)

# Capteurs monochromes stéréo (CAM_B & CAM_C)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_left.setFps(30)
# Optimisations capteur : compensation d'exposition (-3) pour supprimer la saturation blanche
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

# Moteur stéréo matériel (Myriad X)
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)

# --- REGLAGES CLES DE PRECISION ---
# 1. Disparité sous-pixel : indispensable pour une précision continue au millimètre
stereo.setSubpixel(True)

# 2. Extended Disparity : DESACTIVE pour garantir l'activation du filtre médian matériel 7x7
stereo.setExtendedDisparity(False)

# 3. Left-Right Check : actif pour éliminer les occlusions
stereo.setLeftRightCheck(True)

# 4. Alignement pixel-à-pixel sur la caméra couleur
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# 5. Filtre médian matériel KERNEL_7x7
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

# 6. Configuration fine des filtres et seuils
config = stereo.initialConfig.get()
config.postProcessing.decimationFilter.decimationFactor = 1

# Seuil de cohérence gauche-droite plus strict (5 au lieu de 10) pour éviter les bavures sur les arêtes
config.algorithmControl.leftRightCheckThreshold = 5

# Seuil de confiance pour filtrer les faux appariements
config.costMatching.confidenceThreshold = 230

# Filtre spatial très léger (comble les micro-trous sans déborder des arêtes)
config.postProcessing.spatialFilter.enable = False
config.postProcessing.spatialFilter.holeFillingRadius = 1
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.spatialFilter.alpha = 0.35
config.postProcessing.spatialFilter.delta = 8

# Filtre temporel pour stabiliser les mesures
config.postProcessing.temporalFilter.enable = True
config.postProcessing.temporalFilter.persistencyMode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_4
)

# Filtre speckle pour supprimer les pixels isolés
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 40

# Plage de mesure par défaut axée sur l'espace de travail (35 cm à 1.5 m)
min_range = 350
max_range = 1500
config.postProcessing.thresholdFilter.minRange = min_range
config.postProcessing.thresholdFilter.maxRange = max_range

stereo.initialConfig.set(config)

# Liaisons
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

xin_cfg = pipeline.create(dai.node.XLinkIn)
xin_cfg.setStreamName("stereo_cfg")
xin_cfg.out.link(stereo.inputConfig)

# ==============================================================================
# 2. Gestion de la souris, ROI et calculs statistiques
# ==============================================================================
mouse_x, mouse_y = -1, -1
roi_start = None
roi_current = None
roi_active = None
is_drawing_roi = False


def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y, roi_start, roi_current, roi_active, is_drawing_roi

    mouse_x, mouse_y = x, y

    # Clic gauche : début tracé de boîte de mesure (ROI)
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_current = (x, y)
        is_drawing_roi = True

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing_roi:
        roi_current = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and is_drawing_roi:
        is_drawing_roi = False
        if roi_start and roi_current:
            x0, y0 = roi_start
            x1, y1 = roi_current
            if abs(x1 - x0) > 6 and abs(y1 - y0) > 6:
                roi_active = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            else:
                roi_active = None

    # Clic droit : réinitialiser la boîte de mesure
    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_active = None
        roi_start = None
        roi_current = None
        is_drawing_roi = False


def get_median_distance(depth_map, x, y, radius=4, current_min=300, current_max=4000):
    """Mesure ponctuelle adaptative."""
    h, w = depth_map.shape
    if not (0 <= x < w and 0 <= y < h):
        return 0.0, 0.0

    for r in [radius, radius * 2]:
        x1, x2 = max(0, x - r), min(w, x + r + 1)
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        patch = depth_map[y1:y2, x1:x2]
        valid = patch[(patch >= current_min) & (patch <= current_max)]
        if len(valid) >= 5:
            return float(np.median(valid)), float(np.std(valid))

    return 0.0, 0.0


def get_roi_stats(depth_map, x0, y0, x1, y1, current_min=300, current_max=4000):
    """Calcule les statistiques complètes de profondeur dans une zone rectangulaire (ROI)."""
    h, w = depth_map.shape
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x0 >= x1 or y0 >= y1:
        return None

    region = depth_map[y0:y1, x0:x1]
    valid = region[(region >= current_min) & (region <= current_max)]
    if len(valid) < 5:
        return None

    return {
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_count": int(len(valid)),
        "total_count": int(region.size),
        "fill_pct": float(100.0 * len(valid) / region.size),
    }


# ==============================================================================
# 3. Boucle principale d'affichage
# ==============================================================================
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    q_cfg = device.getInputQueue("stereo_cfg")

    win_name = "OAK-D : Mesureur de Distance Haute Precision"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, on_mouse)

    print("\n" + "=" * 65)
    print("  OAK-D DISTANCE VIEWER HAUTE PRECISION (PRO EDITION)")
    print("  - Visez au centre : distance instantanée")
    print("  - Survol souris : distance au curseur")
    print("  - Clic-Glisser gauche : tracer une zone (ROI) pour stats complètes")
    print("  - Clic droit : effacer la zone (ROI)")
    print("  - Touche 'v' : changer de vue (Côte-à-côte | Overlay Superposition | Profondeur)")
    print("  - Touche 'd' : basculer Décimation (1 = Haute Précision / 2 = 30 FPS)")
    print("  - Touche 's' : basculer Filtre Spatial (lissage doux)")
    print("  - Touche '+' / '-' : ajuster la portée (1.2m, 1.5m, 2m, 3.5m, 5m)")
    print("  - Touche 'q' : quitter")
    print("=" * 65 + "\n")

    smooth_dist = 0.0
    decimation_mode = 1
    spatial_enabled = False
    view_modes = ["side_by_side", "overlay", "depth_only"]
    view_idx = 0

    fps_time = time.time()
    fps_counter = 0
    fps_display = 0.0

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()

        rgb_frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()

        h, w = depth_frame.shape
        cx, cy = w // 2, h // 2

        # Calcul FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_display = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        # Masquage et colormap
        valid_mask = (depth_frame >= min_range) & (depth_frame <= max_range)
        depth_clipped = np.clip(depth_frame, min_range, max_range)

        # Normalisation JET : rouge = proche, bleu = limite max_range
        depth_norm = 255 - ((depth_clipped - min_range) / (max_range - min_range) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_color[~valid_mask] = [0, 0, 0]

        # Mode d'affichage
        current_view_mode = view_modes[view_idx]
        display_frame = None

        if current_view_mode == "side_by_side":
            display_frame = np.hstack([rgb_frame.copy(), depth_color.copy()])
            disp_w = 2 * w
        elif current_view_mode == "overlay":
            # Incrustation semi-transparente 50% Depth sur 50% RGB
            overlay = rgb_frame.copy()
            mask_3ch = np.stack([valid_mask] * 3, axis=-1)
            overlay[mask_3ch] = cv2.addWeighted(rgb_frame, 0.45, depth_color, 0.55, 0)[mask_3ch]
            display_frame = overlay
            disp_w = w
        else:  # depth_only
            display_frame = depth_color.copy()
            disp_w = w

        # Conversion des coordonnées de la souris selon la vue
        rx = mouse_x % w if 0 <= mouse_x < disp_w else -1
        ry = mouse_y if 0 <= mouse_y < h else -1

        # Tracé de la boîte ROI en cours de dessin
        if is_drawing_roi and roi_start and roi_current:
            x0, y0 = roi_start
            x1, y1 = roi_current
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Traitement de la ROI active
        if roi_active:
            x0, y0, x1, y1 = roi_active
            # Adapter coordonnées pour calcul sur depth_map
            dx0, dx1 = x0 % w, x1 % w
            if dx0 > dx1:
                dx0, dx1 = dx1, dx0
            stats = get_roi_stats(depth_frame, dx0, y0, dx1, y1, current_min=min_range, current_max=max_range)

            # Dessin de la boîte sur l'affichage
            cv2.rectangle(display_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if current_view_mode == "side_by_side" and x0 < w:
                cv2.rectangle(display_frame, (x0 + w, y0), (x1 + w, y1), (0, 255, 0), 2)

            if stats:
                txt_roi_1 = f"ROI: {stats['median']/1000.0:.2f} m ({int(stats['median'])} mm +/- {int(stats['std'])}mm)"
                txt_roi_2 = f"Min: {stats['min']/1000.0:.2f}m | Max: {stats['max']/1000.0:.2f}m | Rempli: {stats['fill_pct']:.0f}%"
                cv2.rectangle(display_frame, (10, h - 65), (550, h - 10), (0, 0, 0), -1)
                cv2.putText(display_frame, txt_roi_1, (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display_frame, txt_roi_2, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
        else:
            # Mesure stabilisée au centre si aucune ROI n'est active
            raw_center_dist, center_std = get_median_distance(
                depth_frame, cx, cy, radius=4, current_min=min_range, current_max=max_range
            )
            if raw_center_dist > 0:
                smooth_dist = raw_center_dist if smooth_dist == 0.0 else (0.80 * smooth_dist + 0.20 * raw_center_dist)
                txt_center = f"{smooth_dist / 1000.0:.2f} m ({int(smooth_dist)} mm +/- {int(center_std)}mm)"
                txt_color = (0, 255, 0)
            else:
                txt_center = f"Hors portee (<{min_range/1000:.1f}m ou >{max_range/1000:.1f}m)"
                txt_color = (0, 0, 255)

            # Réticules
            if current_view_mode == "side_by_side":
                cv2.drawMarker(display_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                cv2.drawMarker(display_frame, (cx + w, cy), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
            else:
                cv2.drawMarker(display_frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

            # Mesure sous la souris
            if rx >= 0 and ry >= 0 and not is_drawing_roi:
                m_dist, m_std = get_median_distance(
                    depth_frame, rx, ry, radius=4, current_min=min_range, current_max=max_range
                )
                if m_dist > 0:
                    m_txt = f"{m_dist / 1000.0:.2f} m ({int(m_dist)} mm)"
                    cv2.circle(display_frame, (mouse_x, ry), 5, (0, 255, 255), -1)
                    cv2.putText(
                        display_frame, m_txt, (mouse_x + 12, ry - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
                    )

        # Bandeau HUD supérieur
        cv2.rectangle(display_frame, (10, 10), (620, 75), (0, 0, 0), -1)
        if not roi_active:
            cv2.putText(display_frame, f"Distance : {txt_center}", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.68, txt_color, 2)
        else:
            cv2.putText(
                display_frame,
                "Zone ROI active (Clic droit pour effacer)",
                (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

        hud_info = f"FPS: {fps_display:.1f} | Vue: {current_view_mode.upper()} ('v') | Decim: {decimation_mode} | Portee: {max_range/1000:.1f}m"
        cv2.putText(display_frame, hud_info, (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        cv2.imshow(win_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("v"):
            # Bascule de mode d'affichage
            view_idx = (view_idx + 1) % len(view_modes)
            print(f"[INFO] Mode d'affichage : {view_modes[view_idx]}")
        elif key == ord("d"):
            decimation_mode = 2 if decimation_mode == 1 else 1
            cfg = stereo.initialConfig.get()
            cfg.postProcessing.decimationFilter.decimationFactor = decimation_mode
            q_cfg.send(cfg)
            print(f"[INFO] Décimation: {decimation_mode}")
        elif key == ord("s"):
            spatial_enabled = not spatial_enabled
            cfg = stereo.initialConfig.get()
            cfg.postProcessing.spatialFilter.enable = spatial_enabled
            cfg.postProcessing.spatialFilter.holeFillingRadius = 1
            cfg.postProcessing.spatialFilter.numIterations = 1
            q_cfg.send(cfg)
            print(f"[INFO] Filtre spatial: {'ON' if spatial_enabled else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            max_range = min(5000, max_range + 500)
            cfg = stereo.initialConfig.get()
            cfg.postProcessing.thresholdFilter.maxRange = max_range
            q_cfg.send(cfg)
            print(f"[INFO] Portée max: {max_range/1000:.1f} m")
        elif key in (ord("-"), ord("_")):
            max_range = max(1000, max_range - 500)
            cfg = stereo.initialConfig.get()
            cfg.postProcessing.thresholdFilter.maxRange = max_range
            q_cfg.send(cfg)
            print(f"[INFO] Portée max: {max_range/1000:.1f} m")

cv2.destroyAllWindows()
