#!/usr/bin/env python3
"""
Outil de calibration en deux temps de l'homographie de la table pour RobotKraft :
  1. Étape 1 : Sélection manuelle des 4 points de la table à la souris sur l'image caméra.
  2. Étape 2 : Fermeture de la première fenêtre et affichage d'une seconde fenêtre présentant
               le résultat de l'homographie appliqué à l'ensemble de l'image (avec possibilité
               de revenir en arrière pour réajuster ou de sauvegarder).

Utilisation :
  .venv/bin/python calibrate_table.py
  .venv/bin/python calibrate_table.py --image oak_tools/dataset_yolo/images/val/oak_20261003_160122_225_0001.jpg
  .venv/bin/python calibrate_table.py --width-mm 800 --height-mm 600

Touches :
  Étape 1 (Sélection) :
    - Clic gauche         : Placer un point / Glisser-déposer un point
    - Flèches clavier     : Micro-ajustement du point sélectionné (1, 2, 3 ou 4)
    - [u] / [z] / Clic-D  : Annuler le dernier point
    - [r]                 : Réinitialiser tous les points
    - [ESPACE] (live)     : Figer / défiger la vidéo en direct
    - [ENTREE] / [ESPACE] : Valider les 4 points et passer à l'Étape 2
    - [q] / [ESC]         : Quitter

  Étape 2 (Résultat homographie) :
    - [s] / [ENTREE]      : Sauvegarder la calibration (table_homography.json)
    - [b] / [Backspace]   : Revenir à l'Étape 1 pour réajuster les points
    - [t]                 : Basculer vue globale (ensemble image) / vue recadrée table
    - [g]                 : Afficher / masquer la grille
    - [+/-]               : Zoomer / dézoomer sur l'image redressée
    - Clic gauche         : Inspecter les coordonnées d'un point (mm / px)
    - [q] / [ESC]         : Quitter
"""

import argparse
import glob
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
parent_dir = os.path.dirname(SCRIPT_DIR)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from table_homography import TableHomography

CORNER_COLORS = [
    (255, 255, 0),  # P1: Cyan (Haut-Gauche)
    (0, 255, 0),    # P2: Vert (Haut-Droite)
    (0, 215, 255),  # P3: Jaune (Bas-Droite)
    (0, 140, 255),  # P4: Orange (Bas-Gauche)
]
CORNER_NAMES = [
    "P1: Haut-Gauche (TL)",
    "P2: Haut-Droite (TR)",
    "P3: Bas-Droite (BR)",
    "P4: Bas-Gauche (BL)",
]


class TableCalibrator:
    def __init__(
        self,
        source_frame: np.ndarray,
        config_path: str = "table_homography.json",
        table_size_mm: Optional[Tuple[float, float]] = None,
        dst_size_px: Tuple[int, int] = (1920, 1080),
    ):
        self.raw_frame = source_frame.copy()
        self.config_path = config_path
        self.table_size_mm = table_size_mm
        self.dst_size_px = dst_size_px
        self.img_h, self.img_w = self.raw_frame.shape[:2]

        self.points: List[Tuple[float, float]] = []
        self.selected_point_idx: Optional[int] = None
        self.dragging: bool = False
        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.show_grid: bool = True
        self.view_mode: str = "full"  # "full" = ensemble image, "crop" = table seule
        self.zoom: float = 1.0

        # Clic de test dans l'étape 2
        self.result_clicked_pt: Optional[Tuple[int, int]] = None
        self.result_mouse_pos: Tuple[int, int] = (0, 0)

        self.status_message: str = ""
        self.status_time: float = 0.0

        # Charger une calibration existante si présente
        if os.path.exists(self.config_path):
            try:
                homo = TableHomography.load(self.config_path)
                orig_w, orig_h = homo.image_size if homo.image_size else (self.img_w, self.img_h)
                sx = self.img_w / float(orig_w)
                sy = self.img_h / float(orig_h)
                self.points = [(float(pt[0] * sx), float(pt[1] * sy)) for pt in homo.src_points]
                if self.table_size_mm is None and homo.table_width_mm and homo.table_height_mm:
                    self.table_size_mm = (homo.table_width_mm, homo.table_height_mm)
                self.set_status("Points precedents pre-charges ! Vous pouvez les ajuster ou [ENTREE].", 4.0)
            except Exception as e:
                print(f"[Avertissement] Impossible de charger {self.config_path}: {e}")

    def set_status(self, msg: str, duration: float = 3.0):
        self.status_message = msg
        self.status_time = time.time() + duration

    def on_mouse_step1(self, event, x, y, flags, param):
        """Gestionnaire de souris pour l'Étape 1 (Sélection)."""
        self.mouse_pos = (max(0, min(self.img_w - 1, x)), max(0, min(self.img_h - 1, y)))
        click_radius = 20

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_idx = None
            for i, (px, py) in enumerate(self.points):
                if np.hypot(px - x, py - y) < click_radius:
                    clicked_idx = i
                    break

            if clicked_idx is not None:
                self.selected_point_idx = clicked_idx
                self.dragging = True
            elif len(self.points) < 4:
                self.points.append((float(x), float(y)))
                self.selected_point_idx = len(self.points) - 1
                if len(self.points) == 4:
                    self.set_status("4 points definis ! Appuyez sur [ENTREE] pour afficher le resultat.", 5.0)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_point_idx is not None:
                self.points[self.selected_point_idx] = (float(x), float(y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo_point()

    def on_mouse_step2(self, event, x, y, flags, param):
        """Gestionnaire de souris pour l'Étape 2 (Inspection du résultat)."""
        self.result_mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.result_clicked_pt = (x, y)

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.selected_point_idx = len(self.points) - 1 if self.points else None
            self.set_status("Dernier point supprime.")

    def reset_points(self):
        self.points.clear()
        self.selected_point_idx = None
        self.set_status("Points reinitialises. Cliquez pour definir le Point 1.")

    def move_selected_point(self, dx: float, dy: float):
        if self.selected_point_idx is not None and 0 <= self.selected_point_idx < len(self.points):
            px, py = self.points[self.selected_point_idx]
            new_x = max(0.0, min(float(self.img_w - 1), px + dx))
            new_y = max(0.0, min(float(self.img_h - 1), py + dy))
            self.points[self.selected_point_idx] = (new_x, new_y)

    def draw_loupe(self, img: np.ndarray):
        """Loupe de précision (zoom 4x) en haut à droite avec réticule."""
        mx, my = self.mouse_pos
        zoom = 4
        loupe_w, loupe_h = 160, 160
        roi_w = loupe_w // zoom
        roi_h = loupe_h // zoom

        x1 = max(0, min(self.img_w - roi_w, mx - roi_w // 2))
        y1 = max(0, min(self.img_h - roi_h, my - roi_h // 2))
        x2 = x1 + roi_w
        y2 = y1 + roi_h

        crop = self.raw_frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] != roi_h or crop.shape[1] != roi_w:
            return

        zoomed = cv2.resize(crop, (loupe_w, loupe_h), interpolation=cv2.INTER_NEAREST)
        cx, cy = loupe_w // 2, loupe_h // 2
        cv2.line(zoomed, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 1)
        cv2.line(zoomed, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 1)
        cv2.circle(zoomed, (cx, cy), 3, (0, 255, 255), 1)

        pad = 15
        ox = self.img_w - loupe_w - pad
        oy = pad

        cv2.rectangle(img, (ox - 2, oy - 2), (ox + loupe_w + 2, oy + loupe_h + 2), (255, 255, 255), 2)
        cv2.rectangle(img, (ox - 3, oy - 3), (ox + loupe_w + 3, oy + loupe_h + 3), (0, 0, 0), 1)
        img[oy : oy + loupe_h, ox : ox + loupe_w] = zoomed

        coord_txt = f"X:{mx:4d} Y:{my:4d}"
        cv2.rectangle(img, (ox, oy + loupe_h - 22), (ox + loupe_w, oy + loupe_h), (0, 0, 0), -1)
        cv2.putText(img, coord_txt, (ox + 10, oy + loupe_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def draw_hud_step1(self, img: np.ndarray):
        """HUD de l'Étape 1 : Sélection des points."""
        banner_h = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.img_w, banner_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.line(img, (0, banner_h), (self.img_w, banner_h), (100, 100, 100), 1)

        n_pts = len(self.points)
        if n_pts < 4:
            next_name = CORNER_NAMES[n_pts]
            col = CORNER_COLORS[n_pts]
            txt_step = f"ETAPE 1/2 : Selectionnez le coin [{next_name}] ({n_pts + 1}/4)"
        else:
            txt_step = "ETAPE 1/2 TERMINEE : Appuyez sur [ENTREE] pour AFFICHER LE RESULTAT"
            col = (0, 255, 0)

        cv2.putText(img, txt_step, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, txt_step, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 1, cv2.LINE_AA)

        shortcuts = (
            "[Clic-G]: Placer/Deplacer | [Fleches]: Micro-ajust. | [1-4]: Choix point | "
            "[U]: Annuler | [R]: Reset | [ENTREE]: Valider & Afficher Resultat"
        )
        cv2.putText(img, shortcuts, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)

        if time.time() < self.status_time and self.status_message:
            cv2.rectangle(img, (0, self.img_h - 40), (self.img_w, self.img_h), (0, 140, 0), -1)
            cv2.putText(
                img,
                self.status_message,
                (20, self.img_h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def draw_elements_step1(self, img: np.ndarray):
        """Dessine les points et le quadrilatère de l'étape 1."""
        n_pts = len(self.points)
        if n_pts == 4:
            pts_arr = np.array(self.points, dtype=np.int32)
            poly_overlay = img.copy()
            cv2.fillPoly(poly_overlay, [pts_arr], (0, 180, 0))
            cv2.addWeighted(poly_overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [pts_arr], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        elif n_pts > 1:
            for i in range(n_pts - 1):
                p_start = (int(self.points[i][0]), int(self.points[i][1]))
                p_end = (int(self.points[i + 1][0]), int(self.points[i + 1][1]))
                cv2.line(img, p_start, p_end, (0, 255, 255), 2, lineType=cv2.LINE_AA)

        for i, (px, py) in enumerate(self.points):
            ix, iy = int(round(px)), int(round(py))
            col = CORNER_COLORS[i]
            is_selected = (i == self.selected_point_idx)

            if is_selected:
                cv2.circle(img, (ix, iy), 14, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(img, (ix, iy), 16, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.circle(img, (ix, iy), 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (ix, iy), 5, col, -1, cv2.LINE_AA)

            lbl = f"P{i+1}"
            offset_y = -12 if i in [0, 1] else 24
            offset_x = -20 if i in [0, 3] else 10
            cv2.putText(img, lbl, (ix + offset_x, iy + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (ix + offset_x, iy + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)

    def get_homography(self) -> Optional[TableHomography]:
        if len(self.points) != 4:
            return None
        return TableHomography(
            src_points=self.points,
            dst_size_px=self.dst_size_px,
            table_size_mm=self.table_size_mm,
            image_size=(self.img_w, self.img_h),
            zoom=self.zoom,
            metadata={"source_resolution": [self.img_w, self.img_h]},
        )

    def draw_result_frame(self, homo: TableHomography) -> np.ndarray:
        """Génère l'affichage complet de l'Étape 2 : Résultat de l'homographie."""
        warped = homo.warp_image(self.raw_frame, mode=self.view_mode)
        h, w = warped.shape[:2]

        # Grille orthogonale
        if self.show_grid:
            step_x = max(30, w // 10)
            step_y = max(30, h // 8)
            for x in range(0, w, step_x):
                cv2.line(warped, (x, 0), (x, h), (180, 180, 180), 1, cv2.LINE_AA)
            for y in range(0, h, step_y):
                cv2.line(warped, (0, y), (w, y), (180, 180, 180), 1, cv2.LINE_AA)

        # Dessiner le contour de la table en vert
        warped = homo.draw_table_on_warped(warped, mode=self.view_mode, color=(0, 255, 0))

        # Clic d'inspection
        if self.result_clicked_pt:
            cx, cy = self.result_clicked_pt
            if 0 <= cx < w and 0 <= cy < h:
                cv2.drawMarker(warped, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(warped, (cx, cy), 6, (0, 0, 255), -1)

        # Bandeau haut HUD Étape 2
        banner_h = 75
        overlay = warped.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, warped, 0.15, 0, warped)
        cv2.line(warped, (0, banner_h), (w, banner_h), (120, 120, 120), 1)

        mode_str = "IMAGE COMPLETE REDRESSEE" if self.view_mode == "full" else "RECADREE TABLE"
        title_txt = f"ETAPE 2/2 : RESULTAT HOMOGRAPHIE [{mode_str}]"
        cv2.putText(warped, title_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(warped, title_txt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 1, cv2.LINE_AA)

        shortcuts = (
            "[S] ou [ENTREE]: Sauvegarder la calibration | [B]: Revenir a la selection | "
            f"[T]: Basculer vue | [+/-]: Zoom ({self.zoom:.1f}x) | [G]: Grille | [Q]: Quitter"
        )
        cv2.putText(warped, shortcuts, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)

        # Bandeau bas d'informations
        bottom_h = 35
        cv2.rectangle(warped, (0, h - bottom_h), (w, h), (0, 0, 0), -1)
        info_txt = f"Resolution: {w}x{h} px"
        if homo.table_width_mm and homo.table_height_mm:
            info_txt += f" | Table reelle: {homo.table_width_mm:.0f}x{homo.table_height_mm:.0f} mm"

        if self.result_clicked_pt:
            cx, cy = self.result_clicked_pt
            if homo.table_width_mm:
                # Convertir pixel vue redressée vers mm table
                cam_u, cam_v = homo.table_to_pixel(cx, cy, from_mm=False, mode=self.view_mode)
                x_mm, y_mm = homo.pixel_to_table(cam_u, cam_v, in_mm=True)
                info_txt += f" | Dernier clic: X={x_mm:.1f} mm, Y={y_mm:.1f} mm (px: {cx}, {cy})"
            else:
                info_txt += f" | Dernier clic pixel: ({cx}, {cy})"

        cv2.putText(warped, info_txt, (20, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        # Statut temporaire (ex: sauvegarde réussie)
        if time.time() < self.status_time and self.status_message:
            cv2.rectangle(warped, (0, h - 70), (w, h - bottom_h), (0, 150, 0), -1)
            cv2.putText(
                warped,
                self.status_message,
                (20, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return warped

    def save(self):
        homo = self.get_homography()
        if homo is None:
            self.set_status("Erreur : Les 4 points ne sont pas encore tous definis !", 3.0)
            return False

        homo.save(self.config_path)
        msg = f"CALIBRATION ENREGISTREE AVEC SUCCES DANS : {os.path.basename(self.config_path)}"
        self.set_status(msg, 5.0)
        print("\n" + "=" * 65)
        print(f"  [SUCCES] Matrice d'homographie sauvegardee dans :")
        print(f"  --> {os.path.abspath(self.config_path)}")
        print("\n  Points source (camera) :")
        for i, (x, y) in enumerate(homo.src_points):
            print(f"    {CORNER_NAMES[i]} : ({x:.1f}, {y:.1f}) px")
        if homo.table_width_mm and homo.table_height_mm:
            print(f"\n  Dimensions table reelle : {homo.table_width_mm:.1f} x {homo.table_height_mm:.1f} mm")
            print(f"  Echelle : {homo.px_per_mm_x:.3f} px/mm (X), {homo.px_per_mm_y:.3f} px/mm (Y)")
        print("=" * 65 + "\n")
        return True


def find_default_image() -> Optional[str]:
    search_patterns = [
        "oak_tools/dataset_yolo/images/val/oak_*.jpg",
        "oak_tools/dataset_yolo/images/train/oak_*.jpg",
        "oak_tools/dataset_raw/oak_*.jpg",
        "dataset_yolo/images/val/oak_*.jpg",
        "dataset_raw/oak_*.jpg",
    ]
    candidates = []
    for pat in search_patterns:
        candidates.extend(glob.glob(os.path.join(SCRIPT_DIR, "..", pat)))
        candidates.extend(glob.glob(pat))

    if candidates:
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    return None


def open_camera(source_type: str = "auto", device: str = "/dev/video0"):
    if source_type in ("auto", "oak"):
        try:
            import depthai as dai
            devs = dai.Device.getAllAvailableDevices()
            if devs:
                print(f"[Camera] OAK-D detectee ({devs[0].getMxId()}), demarrage du pipeline...")
                pipeline = dai.Pipeline()
                cam_rgb = pipeline.create(dai.node.ColorCamera)
                cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
                cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
                cam_rgb.setFps(30)
                xout = pipeline.create(dai.node.XLinkOut)
                xout.setStreamName("rgb")
                cam_rgb.video.link(xout.input)
                device_dai = dai.Device(pipeline)
                q_rgb = device_dai.getOutputQueue("rgb", maxSize=4, blocking=False)
                return "oak", device_dai, q_rgb
        except Exception as e:
            if source_type == "oak":
                print(f"[Erreur] Echec connexion OAK-D : {e}")

    if source_type in ("auto", "usb"):
        dev_idx = int(device) if device.isdigit() else device
        cap = cv2.VideoCapture(dev_idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"[Camera] Webcam V4L2 ouverte ({device})")
            return "usb", cap, None

    return None, None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration interactive de l'homographie de la table en deux etapes")
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Image fixe pour la calibration (ex: oak_tools/dataset_yolo/images/val/oak_...jpg)",
    )
    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        choices=["auto", "oak", "usb", "none"],
        default="auto",
        help="Source camera en direct (auto, oak, usb, none). Defaut: auto",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="/dev/video0",
        help="Chemin ou index du device V4L2 pour la webcam USB (Defaut: /dev/video0)",
    )
    parser.add_argument(
        "--config",
        "-o",
        type=str,
        default="table_homography.json",
        help="Chemin du fichier de sortie JSON (Defaut: table_homography.json)",
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=None,
        help="Largeur reelle de la table en millimetres (optionnel, ex: 800)",
    )
    parser.add_argument(
        "--height-mm",
        type=float,
        default=None,
        help="Longueur/hauteur reelle de la table en millimetres (optionnel, ex: 600)",
    )
    parser.add_argument(
        "--output-w",
        type=int,
        default=None,
        help="Largeur en pixels de la vue deployee (Defaut: meme que l'image)",
    )
    parser.add_argument(
        "--output-h",
        type=int,
        default=None,
        help="Hauteur en pixels de la vue deployee (Defaut: meme que l'image)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    table_size_mm = None
    if args.width_mm and args.height_mm:
        table_size_mm = (args.width_mm, args.height_mm)

    cam_type = None
    cam_dev = None
    cam_queue = None
    current_frame = None

    if args.image:
        if not os.path.exists(args.image):
            print(f"[Erreur] Image introuvable : {args.image}", file=sys.stderr)
            sys.exit(1)
        current_frame = cv2.imread(args.image)
        print(f"[Source] Image chargee : {args.image} ({current_frame.shape[1]}x{current_frame.shape[0]})")
    elif args.camera != "none":
        cam_type, cam_dev, cam_queue = open_camera(args.camera, args.device)
        if cam_type:
            print(f"[Source] Flux direct active via {cam_type.upper()}")
            if cam_type == "oak":
                current_frame = cam_queue.get().getCvFrame()
            elif cam_type == "usb":
                ret, current_frame = cam_dev.read()

    if current_frame is None:
        auto_img = find_default_image()
        if auto_img and os.path.exists(auto_img):
            print(f"[Source] Aucune camera active detectee. Utilisation de la photo recente : {auto_img}")
            current_frame = cv2.imread(auto_img)
        else:
            print(
                "[Erreur] Impossible d'obtenir une image de la table.\n"
                "Veuillez specifier une photo avec --image <chemin> ou brancher la camera.",
                file=sys.stderr,
            )
            sys.exit(1)

    out_w = args.output_w or current_frame.shape[1]
    out_h = args.output_h or current_frame.shape[0]

    calibrator = TableCalibrator(
        source_frame=current_frame,
        config_path=args.config,
        table_size_mm=table_size_mm,
        dst_size_px=(out_w, out_h),
    )

    win_step1 = "RobotKraft - Etape 1 : Selection des 4 points"
    win_step2 = "RobotKraft - Etape 2 : Resultat Homographie (Image Entiere)"

    # État courant : "step1" (sélection) ou "step2" (résultat)
    current_step = "step1"

    cv2.namedWindow(win_step1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_step1, 1280, 720)
    cv2.setMouseCallback(win_step1, calibrator.on_mouse_step1)

    live_paused = (cam_type is None)

    print("\n" + "=" * 65)
    print("  CALIBRATION HOMOGRAPHIE TABLE (2 ETAPES)")
    print("  " + "-" * 61)
    print("  ETAPE 1 : Cliquez sur les 4 coins de la table fixe a la souris :")
    print("     - P1 : Coin Haut-Gauche (TL)")
    print("     - P2 : Coin Haut-Droite (TR)")
    print("     - P3 : Coin Bas-Droite  (BR)")
    print("     - P4 : Coin Bas-Gauche  (BL)")
    print("  Une fois les 4 points places, appuyez sur [ENTREE] pour passer")
    print("  a l'ETAPE 2 (fermeture de la 1ere fenetre et affichage du resultat).")
    print("=" * 65 + "\n")

    try:
        while True:
            # -------------------------------------------------------------
            # ÉTAPE 1 : SÉLECTION DES 4 POINTS À LA SOURIS
            # -------------------------------------------------------------
            if current_step == "step1":
                if cam_type and not live_paused:
                    if cam_type == "oak":
                        in_frame = cam_queue.tryGet()
                        if in_frame is not None:
                            current_frame = in_frame.getCvFrame()
                            calibrator.raw_frame = current_frame.copy()
                    elif cam_type == "usb":
                        ret, frame = cam_dev.read()
                        if ret:
                            current_frame = frame
                            calibrator.raw_frame = current_frame.copy()

                display_img = calibrator.raw_frame.copy()
                calibrator.draw_elements_step1(display_img)
                calibrator.draw_loupe(display_img)
                calibrator.draw_hud_step1(display_img)

                cv2.imshow(win_step1, display_img)

                raw_key = cv2.waitKey(20)
                key = raw_key & 0xFF

                if key in [ord("q"), 27]:
                    break
                elif key in [10, 13]:  # Entrée -> Passage à l'Étape 2 si 4 points
                    if len(calibrator.points) == 4:
                        # Fermer la fenêtre Étape 1 et ouvrir l'Étape 2
                        cv2.destroyWindow(win_step1)
                        current_step = "step2"
                        cv2.namedWindow(win_step2, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(win_step2, 1280, 720)
                        cv2.setMouseCallback(win_step2, calibrator.on_mouse_step2)
                        calibrator.set_status("Etape 2 : Visualisation du resultat. Appuyez sur [S] pour sauvegarder.", 4.0)
                    else:
                        calibrator.set_status(f"Placez d'abord les 4 points ({len(calibrator.points)}/4 actuellement).", 2.5)
                elif key == ord(" "):
                    if cam_type:
                        live_paused = not live_paused
                        stat = "Flux fige" if live_paused else "Flux en direct"
                        calibrator.set_status(stat, 2.0)
                    elif len(calibrator.points) == 4:
                        # Si déjà 4 points et pas de flux live, espace permet aussi de valider
                        cv2.destroyWindow(win_step1)
                        current_step = "step2"
                        cv2.namedWindow(win_step2, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(win_step2, 1280, 720)
                        cv2.setMouseCallback(win_step2, calibrator.on_mouse_step2)
                elif key in [ord("u"), ord("z")]:
                    calibrator.undo_point()
                elif key == ord("r"):
                    calibrator.reset_points()
                elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
                    idx = int(chr(key)) - 1
                    if idx < len(calibrator.points):
                        calibrator.selected_point_idx = idx
                        calibrator.set_status(f"Point P{idx+1} selectionne pour micro-ajustement.", 2.0)

                step = 1.0
                if raw_key in (81, 65361, ord("h"), ord("H")):
                    calibrator.move_selected_point(-step, 0)
                elif raw_key in (83, 65363, ord("l"), ord("L")):
                    calibrator.move_selected_point(step, 0)
                elif raw_key in (82, 65362, ord("k"), ord("K")):
                    calibrator.move_selected_point(0, -step)
                elif raw_key in (84, 65364, ord("j"), ord("J")):
                    calibrator.move_selected_point(0, step)

            # -------------------------------------------------------------
            # ÉTAPE 2 : AFFICHAGE DU RÉSULTAT DE L'HOMOGRAPHIE
            # -------------------------------------------------------------
            elif current_step == "step2":
                homo = calibrator.get_homography()
                if homo is None:
                    current_step = "step1"
                    continue

                result_img = calibrator.draw_result_frame(homo)
                cv2.imshow(win_step2, result_img)

                raw_key = cv2.waitKey(20)
                key = raw_key & 0xFF

                if key in [ord("q"), 27]:
                    break
                elif key in [ord("s"), ord("S"), 10, 13]:  # Sauvegarder
                    calibrator.save()
                elif key in [ord("b"), ord("B"), 8]:  # 'b' ou Backspace -> Retour Étape 1
                    cv2.destroyWindow(win_step2)
                    current_step = "step1"
                    cv2.namedWindow(win_step1, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_step1, 1280, 720)
                    cv2.setMouseCallback(win_step1, calibrator.on_mouse_step1)
                    calibrator.set_status("Retour a l'etape 1 : Vous pouvez reajuster les 4 points.", 3.0)
                elif key in [ord("t"), ord("T")]:
                    calibrator.view_mode = "crop" if calibrator.view_mode == "full" else "full"
                    stat = "Mode vue : ENSEMBLE DE L'IMAGE" if calibrator.view_mode == "full" else "Mode vue : RECADREE TABLE"
                    calibrator.set_status(stat, 2.0)
                elif key in [ord("+"), ord("=")]:
                    calibrator.zoom = min(2.5, round(calibrator.zoom + 0.1, 2))
                    calibrator.set_status(f"Zoom vue globale : {calibrator.zoom:.1f}x", 1.5)
                elif key in [ord("-"), ord("_")]:
                    calibrator.zoom = max(0.4, round(calibrator.zoom - 0.1, 2))
                    calibrator.set_status(f"Zoom vue globale : {calibrator.zoom:.1f}x", 1.5)
                elif key in [ord("g"), ord("G")]:
                    calibrator.show_grid = not calibrator.show_grid

    finally:
        if cam_type == "usb" and cam_dev is not None:
            cam_dev.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
