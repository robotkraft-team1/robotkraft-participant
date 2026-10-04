#!/usr/bin/env python3
"""
Script de démonstration et vérification de la matrice d'homographie de la table.
- Charge la calibration enregistrée (table_homography.json)
- Affiche la vue caméra avec le contour de la table et la grille métrique
- Affiche la vue dépliée à plat (Bird's Eye View)
- Permet de cliquer sur l'une ou l'autre vue pour tester la conversion de coordonnées (X, Y en mm ou px).

Utilisation :
  .venv/bin/python oak_tools/demo_homography.py
  .venv/bin/python oak_tools/demo_homography.py --image oak_tools/dataset_yolo/images/val/oak_20261003_160122_225_0001.jpg
"""

import argparse
import glob
import os
import sys
from typing import Optional

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from table_homography import TableHomography


def find_default_image() -> Optional[str]:
    patterns = [
        "oak_tools/dataset_yolo/images/val/oak_*.jpg",
        "oak_tools/dataset_yolo/images/train/oak_*.jpg",
        "dataset_yolo/images/val/oak_*.jpg",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(os.path.join(SCRIPT_DIR, "..", pat)))
        candidates.extend(glob.glob(pat))
    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Test et démonstration de la transformation d'homographie")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="table_homography.json",
        help="Chemin de la calibration JSON (Défaut: table_homography.json)",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Image de test (par défaut la plus récente du dataset)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        alt = os.path.join(SCRIPT_DIR, args.config)
        if os.path.exists(alt):
            args.config = alt
        else:
            print(f"[Erreur] Le fichier de configuration '{args.config}' n'existe pas encore.")
            print(f"Veuillez d'abord calibrer la table avec :")
            print(f"  python3 oak_tools/calibrate_table.py")
            sys.exit(1)

    homo = TableHomography.load(args.config)
    print(f"[OK] Calibration chargée depuis : {args.config}")
    print(f"  - Points source caméra : {homo.src_points.tolist()}")
    if homo.table_width_mm and homo.table_height_mm:
        print(f"  - Dimensions métriques : {homo.table_width_mm:.1f} x {homo.table_height_mm:.1f} mm")
        print(f"  - Échelle : {homo.px_per_mm_x:.3f} px/mm")

    img_path = args.image or find_default_image()
    if not img_path or not os.path.exists(img_path):
        print("[Erreur] Aucune image trouvée.", file=sys.stderr)
        sys.exit(1)

    raw_frame = cv2.imread(img_path)
    print(f"[Image] Chargée : {img_path} ({raw_frame.shape[1]}x{raw_frame.shape[0]})")

    win_cam = "Vue Camera - Grille et Table (Cliquez pour tester)"
    win_topdown = "Vue Depliee a Plat (Bird's Eye View)"

    cv2.namedWindow(win_cam, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_cam, 1080, 608)
    cv2.namedWindow(win_topdown, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_topdown, homo.dst_width_px, homo.dst_height_px)

    clicked_cam_point = None
    clicked_topdown_point = None

    def on_cam_mouse(event, x, y, flags, param):
        nonlocal clicked_cam_point, clicked_topdown_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_cam_point = (x, y)
            # Calculer le point correspondant sur la table
            xt, yt = homo.pixel_to_table(x, y, in_mm=False, current_img_size=(raw_frame.shape[1], raw_frame.shape[0]))
            clicked_topdown_point = (int(round(xt)), int(round(yt)))
            if homo.table_width_mm:
                x_mm, y_mm = homo.pixel_to_table(x, y, in_mm=True, current_img_size=(raw_frame.shape[1], raw_frame.shape[0]))
                print(f"[Clic Caméra] Pixel: ({x}, {y}) -> Table: ({x_mm:.1f} mm, {y_mm:.1f} mm)")
            else:
                print(f"[Clic Caméra] Pixel: ({x}, {y}) -> Table vue dessus: ({xt:.1f} px, {yt:.1f} px)")

    def on_topdown_mouse(event, x, y, flags, param):
        nonlocal clicked_cam_point, clicked_topdown_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_topdown_point = (x, y)
            # Calculer le point correspondant sur l'image caméra
            uc, vc = homo.table_to_pixel(x, y, from_mm=False, target_img_size=(raw_frame.shape[1], raw_frame.shape[0]))
            clicked_cam_point = (int(round(uc)), int(round(vc)))
            if homo.table_width_mm:
                x_mm = x / homo.px_per_mm_x
                y_mm = y / homo.px_per_mm_y
                print(f"[Clic Table] Pos: ({x_mm:.1f} mm, {y_mm:.1f} mm) -> Pixel Caméra: ({uc:.1f}, {vc:.1f})")
            else:
                print(f"[Clic Table] Vue dessus: ({x}, {y}) -> Pixel Caméra: ({uc:.1f}, {vc:.1f})")

    cv2.setMouseCallback(win_cam, on_cam_mouse)
    cv2.setMouseCallback(win_topdown, on_topdown_mouse)

    print("\n" + "=" * 60)
    print("  DÉMONSTRATION HOMOGRAPHIE TABLE")
    print("  - Cliquez sur la vue caméra pour projeter sur la table.")
    print("  - Cliquez sur la vue à plat pour reprojeter sur la caméra.")
    print("  - Touche 'q' ou ESC pour quitter.")
    print("=" * 60 + "\n")

    while True:
        # 1. Vue caméra avec table et grille
        cam_vis = homo.draw_table_boundary(raw_frame, color=(0, 255, 0), thickness=2)
        if homo.table_width_mm:
            cam_vis = homo.draw_grid_on_image(cam_vis, step_mm=100.0, color=(255, 200, 0))

        if clicked_cam_point:
            cv2.drawMarker(cam_vis, clicked_cam_point, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(cam_vis, clicked_cam_point, 6, (0, 0, 255), -1)

        # 2. Vue dépliée (ensemble de l'image)
        warped_vis = homo.warp_image(raw_frame, mode="full")
        warped_vis = homo.draw_table_on_warped(warped_vis, mode="full", color=(0, 255, 0))

        # Grille orthogonale sur la vue dépliée
        w, h = homo.dst_width_px, homo.dst_height_px
        for gx in range(0, w, 100):
            cv2.line(warped_vis, (gx, 0), (gx, h), (70, 70, 70), 1)
        for gy in range(0, h, 100):
            cv2.line(warped_vis, (0, gy), (w, gy), (70, 70, 70), 1)

        if clicked_topdown_point:
            cv2.drawMarker(warped_vis, clicked_topdown_point, (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.circle(warped_vis, clicked_topdown_point, 5, (0, 0, 255), -1)

        cv2.imshow(win_cam, cam_vis)
        cv2.imshow(win_topdown, warped_vis)

        key = cv2.waitKey(30) & 0xFF
        if key in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
