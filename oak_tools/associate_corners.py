#!/usr/bin/env python3
"""
RobotKraft - Outil d'association interactive 3D <-> 2D
Associe les positions 3D du palpeur (frames.json) aux coins de l'image (matrice.jpg).

Fonctionnalités :
  - Sélection interactive des coins à la souris (clic, glisser-déposer)
  - Loupe de précision intégrée (PIP) zoom 4x avec réticule au pixel près
  - Micro-ajustement au pixel près via les flèches du clavier
  - Raffinement automatique sous-pixel (touche 'a' - cv2.cornerSubPix)
  - Décalage / Inversion de l'ordre des points sans devoir recliquer (touches 'c' et 'i')
  - Calcul en direct de l'homographie plane (pixels <-> mm robot) et des résidus de reprojection
  - Sonde 3D temps réel : survoler l'image affiche les coordonnées robot estimées
  - Résolution PnP (Perspective-n-Point) avec intrinsèques OAK-D
  - Sauvegarde structurée en JSON + image annotée de vérification

Utilisation :
  ./associate_corners.py
  python3 associate_corners.py --frames frames.json --image matrice.jpg
  python3 associate_corners.py --output matrice_association.json
"""

import argparse
from datetime import datetime
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


# Couleurs BGR pour chaque point (haute visibilité)
POINT_COLORS = [
    (255, 255, 0),    # P0 : Cyan
    (0, 255, 0),      # P1 : Vert fluo
    (0, 165, 255),    # P2 : Orange vif
    (255, 0, 255),    # P3 : Magenta
    (0, 255, 255),    # P4 : Jaune
    (255, 128, 0),    # P5 : Bleu ciel
    (128, 0, 255),    # P6 : Violet
    (0, 200, 100),    # P7 : Emeraude
]

# Intrinsèques par défaut OAK-D Lite (1080p, étalonnage usine lu sur le capteur)
DEFAULT_OAK_K = np.array([
    [1507.80,    0.00,  945.58],
    [   0.00, 1508.47,  552.04],
    [   0.00,    0.00,    1.00]
], dtype=np.float64)


class Point3DFrame:
    """Représente une mesure 3D issue de frames.json."""
    def __init__(self, raw: dict, index: int):
        self.raw = raw
        self.index = raw.get("index", index)
        self.tool = raw.get("tool", "palpeur")
        self.ts = raw.get("ts", "")
        # Coordonnées en mètres et millimètres
        self.x_m = float(raw["x"])
        self.y_m = float(raw["y"])
        self.z_m = float(raw["z"])
        self.x_mm = self.x_m * 1000.0
        self.y_mm = self.y_m * 1000.0
        self.z_mm = self.z_m * 1000.0
        self.yaw = raw.get("yaw")
        self.pitch = raw.get("pitch")
        self.roll = raw.get("roll")
        self.pose = raw.get("pose")

    @property
    def xyz_mm(self) -> np.ndarray:
        return np.array([self.x_mm, self.y_mm, self.z_mm], dtype=np.float64)

    @property
    def xyz_m(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m, self.z_m], dtype=np.float64)


class CornerAssociationCalibrator:
    """Gestionnaire de l'interface interactive et des calculs d'association."""

    def __init__(
        self,
        image_path: str = "matrice.jpg",
        frames_path: str = "frames.json",
        output_path: str = "matrice_association.json",
        annotated_path: Optional[str] = None,
        camera_matrix: Optional[np.ndarray] = None,
    ):
        self.image_path = os.path.abspath(image_path)
        self.frames_path = os.path.abspath(frames_path)
        self.output_path = os.path.abspath(output_path)
        if annotated_path is None:
            base, _ = os.path.splitext(self.output_path)
            self.annotated_path = f"{base}_annotated.jpg"
        else:
            self.annotated_path = os.path.abspath(annotated_path)

        self.camera_matrix = camera_matrix if camera_matrix is not None else DEFAULT_OAK_K

        # Chargement de l'image
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image introuvable : {self.image_path}")
        self.raw_image = cv2.imread(self.image_path)
        if self.raw_image is None:
            raise ValueError(f"Impossible de décoder l'image : {self.image_path}")
        self.img_h, self.img_w = self.raw_image.shape[:2]

        # Chargement des points 3D
        self.frames = self._load_frames(self.frames_path)
        self.n_points = len(self.frames)
        if self.n_points < 3:
            raise ValueError(f"Au moins 3 points 3D sont nécessaires, trouvé : {self.n_points}")

        # Points 2D sélectionnés (None si non encore placé)
        # Liste de [u, v] flottants
        self.points_2d: List[Optional[List[float]]] = [None] * self.n_points

        # État de l'interaction
        self.active_idx: int = 0
        self.selected_idx: Optional[int] = 0
        self.dragging: bool = False
        self.mouse_pos: Tuple[int, int] = (self.img_w // 2, self.img_h // 2)

        # Options d'affichage
        self.show_loupe: bool = True
        self.loupe_zoom: float = 4.0
        self.loupe_size: int = 240
        self.show_grid: bool = True
        self.show_reprojection: bool = True

        # Message de statut temporaire
        self.status_msg: str = "Bienvenue ! Cliquez sur l'image pour placer le point P0."
        self.status_time: float = time.time()
        self.status_duration: float = 4.0

        # Résultats de calibration
        self.H_px_to_mm: Optional[np.ndarray] = None
        self.H_mm_to_px: Optional[np.ndarray] = None
        self.reproj_errors_px: List[float] = []
        self.reproj_errors_mm: List[float] = []
        self.pnp_result: Optional[dict] = None
        self.fitted_plane: Optional[dict] = None

        # Charger une session précédente si le fichier existe
        self._load_existing_associations()

        # Calculer initialement si complet
        self.recompute_calibration()

    def _load_frames(self, path: str) -> List[Point3DFrame]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier frames.json introuvable : {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_frames = data.get("frames", [])
        if not raw_frames:
            raise ValueError(f"Aucune frame trouvée dans {path}")

        frames = [Point3DFrame(rf, idx) for idx, rf in enumerate(raw_frames)]
        print(f"[Association] {len(frames)} positions 3D chargées depuis : {os.path.basename(path)}")
        for f in frames:
            print(f"  P{f.index}: X={f.x_mm:6.1f}mm, Y={f.y_mm:6.1f}mm, Z={f.z_mm:5.1f}mm ({f.tool})")
        return frames

    def _load_existing_associations(self):
        """Précharge les coordonnées 2D si le fichier de sortie existe déjà."""
        if not os.path.exists(self.output_path):
            return
        try:
            with open(self.output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts_data = data.get("points") or data.get("associations") or []
            if len(pts_data) == self.n_points:
                for idx, item in enumerate(pts_data):
                    px = item.get("pixel")
                    if px and len(px) == 2:
                        self.points_2d[idx] = [float(px[0]), float(px[1])]
                self.set_status(f"Associations précédentes chargées depuis {os.path.basename(self.output_path)}")
                # Trouver le premier non défini ou sélectionner P0
                self._update_active_point()
        except Exception as e:
            print(f"[Info] Impossible de charger les points existants : {e}")

    def set_status(self, msg: str, duration: float = 3.5):
        self.status_msg = msg
        self.status_time = time.time()
        self.status_duration = duration

    def _update_active_point(self):
        """Met à jour l'index actif vers le prochain point non placé."""
        for i in range(self.n_points):
            if self.points_2d[i] is None:
                self.active_idx = i
                self.selected_idx = i
                return
        # Si tous sont placés, garder le dernier ou premier sélectionné
        if self.selected_idx is None:
            self.selected_idx = 0
            self.active_idx = 0

    @property
    def all_points_placed(self) -> bool:
        return all(p is not None for p in self.points_2d)

    @property
    def placed_count(self) -> int:
        return sum(1 for p in self.points_2d if p is not None)

    # --------------------------------------------------------------------------
    # Calculs géométriques et étalonnage
    # --------------------------------------------------------------------------
    def recompute_calibration(self):
        """Recalcule l'homographie et le PnP dès que 4 points ou plus sont définis."""
        self.H_px_to_mm = None
        self.H_mm_to_px = None
        self.reproj_errors_px = []
        self.reproj_errors_mm = []
        self.pnp_result = None
        self.fitted_plane = None

        if not self.all_points_placed or self.n_points < 4:
            return

        pts_2d = np.array(self.points_2d, dtype=np.float64)  # (N, 2)
        pts_3d_mm = np.array([f.xyz_mm for f in self.frames], dtype=np.float64)  # (N, 3)

        # 1. Homographie plane (projection X, Y robot)
        # Note : les 4 points mesurés ont Z quasi-constant (~40mm)
        pts_3d_xy = pts_3d_mm[:, :2]

        H_px2mm, _ = cv2.findHomography(pts_2d, pts_3d_xy, cv2.RANSAC if len(pts_2d) > 4 else 0)
        if H_px2mm is not None:
            self.H_px_to_mm = H_px2mm
            try:
                self.H_mm_to_px = np.linalg.inv(H_px2mm)
            except np.linalg.LinAlgError:
                self.H_mm_to_px = None

            # Calcul des résidus
            if self.H_mm_to_px is not None:
                # Reprojection 3D -> 2D
                homo_3d = np.column_stack([pts_3d_xy, np.ones(len(pts_3d_xy))])
                pred_2d = (self.H_mm_to_px @ homo_3d.T).T
                pred_2d = pred_2d[:, :2] / pred_2d[:, 2:]
                err_px = np.linalg.norm(pred_2d - pts_2d, axis=1)
                self.reproj_errors_px = [float(e) for e in err_px]

                # Reprojection 2D -> 3D
                homo_2d = np.column_stack([pts_2d, np.ones(len(pts_2d))])
                pred_3d = (self.H_px_to_mm @ homo_2d.T).T
                pred_3d = pred_3d[:, :2] / pred_3d[:, 2:]
                err_mm = np.linalg.norm(pred_3d - pts_3d_xy, axis=1)
                self.reproj_errors_mm = [float(e) for e in err_mm]

        # 2. Plan 3D par moindres carrés (SVD)
        centroid = pts_3d_mm.mean(axis=0)
        centered = pts_3d_mm - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]  # Vecteur propre correspondant à la plus petite valeur
        if normal[2] < 0:
            normal = -normal
        d_plane = -float(np.dot(normal, centroid))
        self.fitted_plane = {
            "normal": [float(n) for n in normal],
            "d": d_plane,
            "centroid_mm": [float(c) for c in centroid],
            "mean_z_mm": float(centroid[2]),
        }

        # 3. PnP (Estimation de la pose de la caméra par rapport au robot)
        if self.camera_matrix is not None:
            dist_coeffs = np.zeros(5, dtype=np.float64)
            # Avec 4 points coplanaires, SQPNP ou EPNP fonctionne de façon robuste
            success = False
            rvec, tvec = None, None
            for flag in [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]:
                try:
                    success, rvec, tvec = cv2.solvePnP(pts_3d_mm, pts_2d, self.camera_matrix, dist_coeffs, flags=flag)
                    if success:
                        break
                except Exception:
                    continue

            if success and rvec is not None and tvec is not None:
                R_mat, _ = cv2.Rodrigues(rvec)
                # Position de la caméra dans le repère robot : C = -R^T * t
                cam_in_robot = -R_mat.T @ tvec.reshape(3, 1)
                self.pnp_result = {
                    "rvec": [float(x) for x in rvec.flatten()],
                    "tvec_mm": [float(x) for x in tvec.flatten()],
                    "rotation_matrix": R_mat.tolist(),
                    "camera_in_robot_mm": [float(x) for x in cam_in_robot.flatten()],
                }

    def pixel_to_robot_mm(self, u: float, v: float) -> Optional[Tuple[float, float, float]]:
        """Convertit un pixel image (u, v) en coordonnées estimées du robot (X, Y, Z) en mm."""
        if self.H_px_to_mm is None:
            return None
        vec = np.array([u, v, 1.0], dtype=np.float64)
        p = self.H_px_to_mm @ vec
        if abs(p[2]) < 1e-9:
            return None
        x = p[0] / p[2]
        y = p[1] / p[2]
        # Estimation de Z sur le plan
        if self.fitted_plane is not None and abs(self.fitted_plane["normal"][2]) > 1e-4:
            nx, ny, nz = self.fitted_plane["normal"]
            d = self.fitted_plane["d"]
            z = -(nx * x + ny * y + d) / nz
        else:
            z = float(np.mean([f.z_mm for f in self.frames]))
        return (x, y, z)

    # --------------------------------------------------------------------------
    # Événements Souris et Clavier
    # --------------------------------------------------------------------------
    def on_mouse(self, event: int, x: int, y: int, flags: int, param):
        """Gestionnaire de clics et mouvements souris."""
        self.mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            # Vérifier si on a cliqué sur un point déjà placé (seuil 22 pixels)
            clicked_idx = self._find_point_near(x, y, radius=22)
            if clicked_idx is not None:
                self.selected_idx = clicked_idx
                self.active_idx = clicked_idx
                self.dragging = True
                self.set_status(f"Point P{clicked_idx} sélectionné (Glisser pour ajuster)")
            else:
                # Placer le point actif à la position du curseur
                idx_to_place = self.active_idx
                self.points_2d[idx_to_place] = [float(x), float(y)]
                self.selected_idx = idx_to_place
                self.dragging = True
                self.set_status(f"Point P{idx_to_place} placé en ({x}, {y})")

                # Passer automatiquement au prochain point non placé
                self._advance_active_to_next_unplaced()
                self.recompute_calibration()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_idx is not None:
                self.points_2d[self.selected_idx] = [float(x), float(y)]
                self.recompute_calibration()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                self.recompute_calibration()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Clic droit : annuler / supprimer le point sous le curseur ou le dernier
            clicked_idx = self._find_point_near(x, y, radius=25)
            if clicked_idx is not None:
                self.points_2d[clicked_idx] = None
                self.active_idx = clicked_idx
                self.selected_idx = clicked_idx
                self.set_status(f"Point P{clicked_idx} supprimé")
                self.recompute_calibration()
            else:
                self.undo_last_point()

    def _find_point_near(self, x: int, y: int, radius: int = 20) -> Optional[int]:
        """Retourne l'index du point le plus proche si dans le rayon."""
        min_dist = float("inf")
        best_idx = None
        for i, pt in enumerate(self.points_2d):
            if pt is not None:
                d = math.hypot(pt[0] - x, pt[1] - y)
                if d <= radius and d < min_dist:
                    min_dist = d
                    best_idx = i
        return best_idx

    def _advance_active_to_next_unplaced(self):
        """Avance active_idx au point non défini suivant."""
        for offset in range(1, self.n_points + 1):
            cand = (self.active_idx + offset) % self.n_points
            if self.points_2d[cand] is None:
                self.active_idx = cand
                return

    def nudge_selected(self, dx: float, dy: float):
        """Micro-déplacement du point actuellement sélectionné."""
        if self.selected_idx is None or self.points_2d[self.selected_idx] is None:
            return
        pt = self.points_2d[self.selected_idx]
        pt[0] = max(0.0, min(float(self.img_w - 1), pt[0] + dx))
        pt[1] = max(0.0, min(float(self.img_h - 1), pt[1] + dy))
        self.recompute_calibration()
        self.set_status(f"P{self.selected_idx} ajusté en ({pt[0]:.1f}, {pt[1]:.1f})")

    def undo_last_point(self):
        """Annule le dernier point placé."""
        for i in reversed(range(self.n_points)):
            if self.points_2d[i] is not None:
                self.points_2d[i] = None
                self.active_idx = i
                self.selected_idx = i
                self.set_status(f"Point P{i} effacé")
                self.recompute_calibration()
                return
        self.set_status("Aucun point à effacer")

    def reset_all(self):
        """Réinitialise tous les points 2D."""
        self.points_2d = [None] * self.n_points
        self.active_idx = 0
        self.selected_idx = 0
        self.recompute_calibration()
        self.set_status("Tous les points ont été réinitialisés !")

    def cycle_order(self, forward: bool = True):
        """Décalage circulaire de l'association (P0->P1, P1->P2, etc.)."""
        if not self.all_points_placed:
            self.set_status("Placez tous les points avant de décaler l'ordre")
            return
        if forward:
            self.points_2d = [self.points_2d[-1]] + self.points_2d[:-1]
            self.set_status("Ordre des points décalé vers l'avant (P0 -> P1)")
        else:
            self.points_2d = self.points_2d[1:] + [self.points_2d[0]]
            self.set_status("Ordre des points décalé vers l'arrière")
        self.recompute_calibration()

    def invert_order(self):
        """Inverse l'ordre des points (sens horaire <-> trigonométrique)."""
        if not self.all_points_placed:
            self.set_status("Placez tous les points avant d'inverser l'ordre")
            return
        self.points_2d.reverse()
        self.recompute_calibration()
        self.set_status("Ordre des points inversé !")

    def snap_subpixel(self):
        """Raffine la position du coin sélectionné avec cv2.cornerSubPix."""
        if self.selected_idx is None or self.points_2d[self.selected_idx] is None:
            self.set_status("Aucun point sélectionné pour le raffinement")
            return
        pt = self.points_2d[self.selected_idx]
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        corners = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        new_x, new_y = float(refined[0][0][0]), float(refined[0][0][1])
        dx = new_x - pt[0]
        dy = new_y - pt[1]
        self.points_2d[self.selected_idx] = [new_x, new_y]
        self.recompute_calibration()
        self.set_status(f"P{self.selected_idx} affiné sous-pixel (Δx={dx:+.2f}, Δy={dy:+.2f})")

    # --------------------------------------------------------------------------
    # Dessin et Rendu de l'Interface
    # --------------------------------------------------------------------------
    def render_frame(self) -> np.ndarray:
        """Génère l'affichage complet de la fenêtre."""
        canvas = self.raw_image.copy()

        # 1. Polygone / quadrilatère si plusieurs points sont placés
        self._draw_polygon_and_edges(canvas)

        # 2. Points placés et cibles
        self._draw_points(canvas)

        # 3. Réticule et mesure sous le curseur souris
        self._draw_cursor_probe(canvas)

        # 4. Loupe de précision (Picture in Picture)
        if self.show_loupe:
            self._draw_loupe_pip(canvas)

        # 5. Bandeau supérieur (instructions) et panneau latéral
        self._draw_hud(canvas)

        # 6. Bandeau inférieur (raccourcis)
        self._draw_footer(canvas)

        return canvas

    def _draw_polygon_and_edges(self, canvas: np.ndarray):
        """Dessine les segments entre points consécutifs et les distances 3D."""
        placed_indices = [i for i, p in enumerate(self.points_2d) if p is not None]
        if len(placed_indices) < 2:
            return

        # Si tous sont placés, dessiner le polygone fermé avec remplissage translucide
        if self.all_points_placed and self.n_points >= 3:
            pts_int = np.array(self.points_2d, dtype=np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts_int], (0, 200, 255))
            cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
            cv2.polylines(canvas, [pts_int], isClosed=True, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

        # Lignes et étiquettes de distance entre points consécutifs
        for i in range(self.n_points):
            next_i = (i + 1) % self.n_points
            p1 = self.points_2d[i]
            p2 = self.points_2d[next_i]

            if p1 is not None and p2 is not None:
                # Segment
                pt1 = (int(round(p1[0])), int(round(p1[1])))
                pt2 = (int(round(p2[0])), int(round(p2[1])))

                # Si ce n'est pas fermé et c'est le dernier segment, ne pas tracer si non complet
                if next_i == 0 and not self.all_points_placed:
                    continue

                if not self.all_points_placed:
                    cv2.line(canvas, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

                # Distance 3D réelle en mm entre ces 2 points
                f1, f2 = self.frames[i], self.frames[next_i]
                d3d_mm = float(np.linalg.norm(f1.xyz_mm - f2.xyz_mm))
                d2d_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

                # Affichage du badge au milieu du segment
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)
                lbl = f"{d3d_mm:.1f}mm ({int(d2d_px)}px)"
                self._draw_text_badge(canvas, lbl, (mid_x, mid_y), (30, 30, 30), (220, 255, 255), scale=0.42, center=True)

    def _draw_points(self, canvas: np.ndarray):
        """Dessine les points sur l'image avec anneaux, croix et badges."""
        for i, pt in enumerate(self.points_2d):
            if pt is None:
                continue

            x, y = int(round(pt[0])), int(round(pt[1]))
            color = POINT_COLORS[i % len(POINT_COLORS)]
            is_selected = (i == self.selected_idx)

            # Anneau extérieur
            outer_radius = 16 if is_selected else 11
            thick = 3 if is_selected else 2
            cv2.circle(canvas, (x, y), outer_radius, (0, 0, 0), thick + 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), outer_radius, color, thick, cv2.LINE_AA)

            # Réticule central
            cross_sz = 8 if is_selected else 6
            cv2.line(canvas, (x - cross_sz, y), (x + cross_sz, y), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(canvas, (x, y - cross_sz), (x, y + cross_sz), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(canvas, (x - cross_sz, y), (x + cross_sz, y), color, 1, cv2.LINE_AA)
            cv2.line(canvas, (x, y - cross_sz), (x, y + cross_sz), color, 1, cv2.LINE_AA)

            # Point central
            cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

            # Badge avec coordonnées 3D
            f = self.frames[i]
            badge_text = f"P{i} [{f.x_mm:.0f}, {f.y_mm:.0f}, {f.z_mm:.0f}mm]"
            offset_y = -22 if (y > 40) else 25
            self._draw_text_badge(canvas, badge_text, (x, y + offset_y), color, (0, 0, 0), scale=0.48, center=True)

    def _draw_cursor_probe(self, canvas: np.ndarray):
        """Affiche les coordonnées 3D sous le curseur souris en mode inspection."""
        mx, my = self.mouse_pos
        if not (0 <= mx < self.img_w and 0 <= my < self.img_h):
            return

        # Réticule discret sous le curseur
        cv2.line(canvas, (mx - 10, my), (mx + 10, my), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (mx, my - 10), (mx, my + 10), (255, 255, 255), 1, cv2.LINE_AA)

        # Si l'homographie est calculée, afficher la position robot prédite sous le curseur
        if self.H_px_to_mm is not None:
            pred = self.pixel_to_robot_mm(mx, my)
            if pred is not None:
                rx, ry, rz = pred
                info_str = f"Robot : X={rx:6.1f}mm, Y={ry:6.1f}mm, Z={rz:5.1f}mm"
                # Positionner le texte à droite du curseur ou à gauche si trop près du bord
                bx = mx + 20 if (mx < self.img_w - 320) else mx - 320
                by = my - 15 if (my > 50) else my + 25
                self._draw_text_badge(canvas, info_str, (bx, by), (20, 20, 20), (0, 255, 200), scale=0.48)

    def _draw_loupe_pip(self, canvas: np.ndarray):
        """Dessine la loupe de précision 4x dans le coin supérieur."""
        mx, my = self.mouse_pos
        sz = self.loupe_size
        zoom = self.loupe_zoom

        # Choisir le coin opposé au curseur pour ne jamais gêner
        if mx > (self.img_w - sz - 40) and my < (sz + 120):
            corner_x = 20
            corner_y = 75
        else:
            corner_x = self.img_w - sz - 20
            corner_y = 75

        # Zone source autour du curseur
        half_src = int(sz / (2 * zoom))
        x1 = max(0, mx - half_src)
        y1 = max(0, my - half_src)
        x2 = min(self.img_w, mx + half_src)
        y2 = min(self.img_h, my + half_src)

        crop = self.raw_image[y1:y2, x1:x2]
        if crop.size == 0:
            return

        loupe_img = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_NEAREST)

        # Réticule central de la loupe
        cx, cy = sz // 2, sz // 2
        cv2.line(loupe_img, (cx - 15, cy), (cx + 15, cy), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(loupe_img, (cx, cy - 15), (cx, cy + 15), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(loupe_img, (cx, cy), 3, (0, 255, 255), 1, cv2.LINE_AA)

        # Cadre noir et blanc autour de la loupe
        cv2.rectangle(loupe_img, (0, 0), (sz - 1, sz - 1), (255, 255, 255), 2)
        cv2.rectangle(loupe_img, (2, 2), (sz - 3, sz - 3), (0, 0, 0), 1)

        # Bandeau de titre de la loupe
        cv2.rectangle(loupe_img, (0, 0), (sz, 22), (30, 30, 30), -1)
        cv2.putText(
            loupe_img,
            f"LOUPE {zoom:.0f}x : u={mx}, v={my}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Incrustation sur le canvas
        canvas[corner_y:corner_y + sz, corner_x:corner_x + sz] = loupe_img

    def _draw_hud(self, canvas: np.ndarray):
        """Dessine le bandeau supérieur et le panneau latéral récapitulatif."""
        # --- 1. Bandeau supérieur (Hauteur 60px) ---
        overlay = canvas[:60, :].copy()
        cv2.rectangle(overlay, (0, 0), (self.img_w, 60), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, canvas[:60, :], 0.15, 0, canvas[:60, :])
        cv2.line(canvas, (0, 60), (self.img_w, 60), (0, 200, 255), 2)

        title = "ROBOTKRAFT - ASSOCIATION 3D (frames.json) <-> COINS 2D (matrice.jpg)"
        cv2.putText(canvas, title, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Instruction dynamique
        if not self.all_points_placed:
            f = self.frames[self.active_idx]
            color = POINT_COLORS[self.active_idx % len(POINT_COLORS)]
            inst = f"Cliquez sur l'image pour placer le POINT P{self.active_idx} (X={f.x_mm:5.1f}mm, Y={f.y_mm:5.1f}mm, Z={f.z_mm:4.1f}mm)"
            cv2.putText(canvas, inst, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        else:
            inst = "Tous les points sont associes ! Appuyez sur [S] ou [Entree] pour sauvegarder."
            cv2.putText(canvas, inst, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

        # Message de statut temporaire à droite
        if (time.time() - self.status_time) < self.status_duration:
            cv2.putText(canvas, f"[i] {self.status_msg}", (self.img_w - 700, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        # --- 2. Panneau latéral gauche (Points & Métriques) ---
        panel_w = 420
        panel_h = 110 + self.n_points * 42 + 90
        panel_x = 20
        panel_y = 75

        # Boîte semi-transparente
        sub = canvas[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        p_overlay = sub.copy()
        cv2.rectangle(p_overlay, (0, 0), (panel_w, panel_h), (25, 25, 25), -1)
        cv2.addWeighted(p_overlay, 0.85, sub, 0.15, 0, sub)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)

        # Titre du panneau
        cv2.putText(canvas, "POINTS 3D (frames.json)", (panel_x + 12, panel_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (panel_x + 10, panel_y + 32), (panel_x + panel_w - 10, panel_y + 32), (80, 80, 80), 1)

        # Liste des points
        curr_y = panel_y + 55
        for i, f in enumerate(self.frames):
            color = POINT_COLORS[i % len(POINT_COLORS)]
            is_active = (i == self.active_idx)
            is_selected = (i == self.selected_idx)
            pt_2d = self.points_2d[i]

            # Indicateur actif '>'
            prefix = " > " if is_active else "   "
            cv2.putText(canvas, prefix, (panel_x + 2, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            # Badge nom
            cv2.putText(canvas, f"P{i}", (panel_x + 28, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

            # 3D
            xyz_str = f"3D: [{f.x_mm:5.0f}, {f.y_mm:5.0f}, {f.z_mm:4.0f}]"
            cv2.putText(canvas, xyz_str, (panel_x + 65, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            # 2D
            if pt_2d is not None:
                err_str = ""
                if i < len(self.reproj_errors_px):
                    err_str = f"e={self.reproj_errors_px[i]:.1f}px"
                px_str = f"2D: ({int(pt_2d[0])}, {int(pt_2d[1])}) {err_str}"
                cv2.putText(canvas, px_str, (panel_x + 235, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(canvas, "[A DEFINIR - Clic]", (panel_x + 235, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1, cv2.LINE_AA)

            curr_y += 38

        # Séparateur métriques
        cv2.line(canvas, (panel_x + 10, curr_y), (panel_x + panel_w - 10, curr_y), (80, 80, 80), 1)
        curr_y += 20

        # Résumé géométrique / calibration
        if self.H_px_to_mm is not None:
            mean_err_px = float(np.mean(self.reproj_errors_px))
            mean_err_mm = float(np.mean(self.reproj_errors_mm))
            cv2.putText(canvas, "HOMOGRAPHIE CALCULEE : OK", (panel_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 2, cv2.LINE_AA)
            curr_y += 20
            cv2.putText(canvas, f"Erreur moy. reprojection : {mean_err_px:.2f} px ({mean_err_mm:.2f} mm)", (panel_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            curr_y += 20
            if self.pnp_result is not None:
                cam_pos = self.pnp_result["camera_in_robot_mm"]
                cv2.putText(canvas, f"Pos Camera robot : [{cam_pos[0]:.0f}, {cam_pos[1]:.0f}, {cam_pos[2]:.0f}] mm", (panel_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        else:
            prog_str = f"Progression : {self.placed_count} / {self.n_points} points places"
            cv2.putText(canvas, prog_str, (panel_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 200, 0), 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray):
        """Dessine le bandeau d'aide inférieur avec les raccourcis clavier."""
        bar_h = 42
        y0 = self.img_h - bar_h
        overlay = canvas[y0:, :].copy()
        cv2.rectangle(overlay, (0, 0), (self.img_w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.90, canvas[y0:, :], 0.10, 0, canvas[y0:, :])
        cv2.line(canvas, (0, y0), (self.img_w, y0), (80, 80, 80), 1)

        shortcuts = (
            "[Clic-G] Placer/Glisser | [1-4] Choisir Pt | [Fleches] Ajuster (Shift: x5) | "
            "[a] Affiner sous-pixel | [u/z] Annuler | [c] Decaler | [i] Inverser | [l] Loupe | "
            "[r] Reset | [s/Entree] Sauvegarder | [q/Esc] Quitter"
        )
        cv2.putText(canvas, shortcuts, (20, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 255), 1, cv2.LINE_AA)

    def _draw_text_badge(
        self,
        canvas: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int],
        text_color: Tuple[int, int, int],
        scale: float = 0.45,
        thickness: int = 1,
        center: bool = False,
    ):
        """Dessine un texte avec une boîte de fond contrastée."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        if center:
            x -= w // 2
            y -= h // 2

        # Clamping dans l'image
        x = max(2, min(self.img_w - w - 4, x))
        y = max(h + 2, min(self.img_h - 4, y))

        cv2.rectangle(canvas, (x - 4, y - h - 4), (x + w + 4, y + baseline + 2), bg_color, -1)
        cv2.rectangle(canvas, (x - 4, y - h - 4), (x + w + 4, y + baseline + 2), (0, 0, 0), 1)
        cv2.putText(canvas, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

    # --------------------------------------------------------------------------
    # Sauvegarde et Exportation
    # --------------------------------------------------------------------------
    def save(self) -> dict:
        """Sauvegarde les résultats d'association en fichier JSON et l'image annotée."""
        if not self.all_points_placed:
            self.set_status("Erreur : veuillez placer tous les points avant de sauvegarder")
            return {}

        self.recompute_calibration()

        points_export = []
        for i, f in enumerate(self.frames):
            pt_2d = self.points_2d[i]
            points_export.append({
                "index": f.index,
                "pixel": [float(pt_2d[0]), float(pt_2d[1])],
                "robot_position_m": [f.x_m, f.y_m, f.z_m],
                "robot_position_mm": [f.x_mm, f.y_mm, f.z_mm],
                "reprojection_error_px": float(self.reproj_errors_px[i]) if i < len(self.reproj_errors_px) else None,
                "reprojection_error_mm": float(self.reproj_errors_mm[i]) if i < len(self.reproj_errors_mm) else None,
                "frame_metadata": {
                    "tool": f.tool,
                    "ts": f.ts,
                    "yaw": f.yaw,
                    "pitch": f.pitch,
                    "roll": f.roll,
                    "pose": f.pose,
                },
            })

        data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "description": "Association 3D (frames.json) <-> 2D (matrice.jpg)",
                "image_path": os.path.basename(self.image_path),
                "image_size": [self.img_w, self.img_h],
                "frames_path": os.path.basename(self.frames_path),
                "num_points": self.n_points,
                "tool": self.frames[0].tool if self.frames else "palpeur",
            },
            "points": points_export,
            "homography_pixel_to_robot_mm": self.H_px_to_mm.tolist() if self.H_px_to_mm is not None else None,
            "homography_robot_mm_to_pixel": self.H_mm_to_px.tolist() if self.H_mm_to_px is not None else None,
            "reprojection_stats": {
                "mean_error_px": float(np.mean(self.reproj_errors_px)) if self.reproj_errors_px else None,
                "max_error_px": float(np.max(self.reproj_errors_px)) if self.reproj_errors_px else None,
                "mean_error_mm": float(np.mean(self.reproj_errors_mm)) if self.reproj_errors_mm else None,
                "max_error_mm": float(np.max(self.reproj_errors_mm)) if self.reproj_errors_mm else None,
            },
            "fitted_plane": self.fitted_plane,
            "pnp_camera_pose": self.pnp_result,
        }

        # Écriture du fichier JSON
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Sauvegarde de l'image annotée de vérification
        annotated = self.render_frame()
        cv2.imwrite(self.annotated_path, annotated)

        self.set_status(f"Succès ! Sauvegardé dans : {os.path.basename(self.output_path)}", duration=6.0)
        print("\n" + "=" * 65)
        print(f"  CALIBRATION SAUVEGARDEE AVEC SUCCES !")
        print(f"  - Fichier JSON  : {self.output_path}")
        print(f"  - Image annotée : {self.annotated_path}")
        if self.reproj_errors_px:
            print(f"  - Erreur moyenne : {np.mean(self.reproj_errors_px):.2f} px ({np.mean(self.reproj_errors_mm):.2f} mm)")
            print(f"  - Erreur max     : {np.max(self.reproj_errors_px):.2f} px ({np.max(self.reproj_errors_mm):.2f} mm)")
        if self.pnp_result:
            cam = self.pnp_result["camera_in_robot_mm"]
            print(f"  - Caméra (repère robot) : X={cam[0]:.1f}mm, Y={cam[1]:.1f}mm, Z={cam[2]:.1f}mm")
        print("=" * 65 + "\n")

        return data

    # --------------------------------------------------------------------------
    # Boucle Principale Interactive
    # --------------------------------------------------------------------------
    def run(self):
        """Lance l'interface graphique interactive."""
        win_name = "RobotKraft - Association 3D Frames <-> Coins Matrice"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, min(1600, self.img_w), min(900, self.img_h))
        cv2.setMouseCallback(win_name, self.on_mouse)

        print("\n" + "=" * 65)
        print("  ASSOCIATION 3D <-> 2D ROBOTKRAFT ACTIVE")
        print("  - Clic gauche sur l'image pour placer successivement les 4 coins.")
        print("  - Les étiquettes de distance s'affichent automatiquement.")
        print("  - Glisser-déposer pour ajuster n'importe quel point.")
        print("  - Flèches clavier pour micro-ajustement au pixel près.")
        print("  - Touche 's' ou [Entrée] pour sauvegarder.")
        print("  - Touche 'q' ou [Echap] pour quitter.")
        print("=" * 65 + "\n")

        while True:
            frame = self.render_frame()
            cv2.imshow(win_name, frame)

            # waitKeyEx pour supporter les flèches du clavier sur tous les OS
            key = cv2.waitKeyEx(25)
            if key == -1:
                continue

            # Standardiser le code touche
            code = key & 0xFF

            # Quitter (q ou Echap)
            if code in (ord("q"), ord("Q"), 27):
                break

            # Sauvegarder (s ou Entrée)
            elif code in (ord("s"), ord("S"), 10, 13):
                self.save()

            # Annuler dernier point (u, z, Backspace)
            elif code in (ord("u"), ord("U"), ord("z"), ord("Z"), 8, 127, 65288):
                self.undo_last_point()

            # Réinitialiser tous les points (r)
            elif code in (ord("r"), ord("R")):
                self.reset_all()

            # Décaler l'ordre cyclique des points (c)
            elif code in (ord("c"), ord("C")):
                self.cycle_order(forward=True)

            # Inverser l'ordre des points (i)
            elif code in (ord("i"), ord("I")):
                self.invert_order()

            # Raffiner sous-pixel (a)
            elif code in (ord("a"), ord("A")):
                self.snap_subpixel()

            # Basculer Loupe PIP (l)
            elif code in (ord("l"), ord("L")):
                self.show_loupe = not self.show_loupe
                self.set_status(f"Loupe {'activée' if self.show_loupe else 'désactivée'}")

            # Sélection directe du point par numéro (0, 1, 2, 3...)
            elif ord("0") <= code <= ord("9"):
                target_idx = code - ord("0")
                if target_idx < self.n_points:
                    self.active_idx = target_idx
                    self.selected_idx = target_idx
                    self.set_status(f"Point P{target_idx} sélectionné")

            # Touches 1-9 du pavé numérique
            elif 65456 <= key <= 65465:
                target_idx = key - 65456
                if target_idx < self.n_points:
                    self.active_idx = target_idx
                    self.selected_idx = target_idx
                    self.set_status(f"Point P{target_idx} sélectionné")

            # Tabulation : point suivant
            elif code in (9, 65289):
                self.active_idx = (self.active_idx + 1) % self.n_points
                self.selected_idx = self.active_idx
                self.set_status(f"Point P{self.active_idx} sélectionné")

            # Flèches directionnelles (micro-ajustement)
            # Linux X11 : 65361=Gauche, 65362=Haut, 65363=Droite, 65364=Bas
            # Windows/autres : 2424832, 2490368, 2555904, 2621440 ou codes 81-84
            elif key in (65361, 2424832, 81):  # Gauche
                self.nudge_selected(-1.0, 0.0)
            elif key in (65363, 2555904, 83):  # Droite
                self.nudge_selected(1.0, 0.0)
            elif key in (65362, 2490368, 82):  # Haut
                self.nudge_selected(0.0, -1.0)
            elif key in (65364, 2621440, 84):  # Bas
                self.nudge_selected(0.0, 1.0)

        cv2.destroyAllWindows()


# ------------------------------------------------------------------------------
# Fonctions utilitaires d'utilisation programmatique (importable)
# ------------------------------------------------------------------------------
def load_association(json_path: str = "matrice_association.json") -> dict:
    """Charge la calibration d'association depuis le fichier JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Fichier de calibration introuvable : {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_pixel_to_robot(
    u: float,
    v: float,
    calib: Union[str, dict] = "matrice_association.json",
) -> Tuple[float, float, float]:
    """
    Convertit un pixel caméra (u, v) en coordonnées robot réelles (X, Y, Z) en mm.

    Args:
        u: Coordonnée X en pixels dans l'image.
        v: Coordonnée Y en pixels dans l'image.
        calib: Chemin vers le JSON ou dictionnaire chargé.

    Returns:
        (X_mm, Y_mm, Z_mm) dans le repère du robot.
    """
    if isinstance(calib, str):
        calib = load_association(calib)

    H_mat = np.array(calib["homography_pixel_to_robot_mm"], dtype=np.float64)
    vec = np.array([u, v, 1.0], dtype=np.float64)
    res = H_mat @ vec
    if abs(res[2]) < 1e-9:
        raise ValueError("Division par zéro lors de la projection d'homographie.")
    x_mm = float(res[0] / res[2])
    y_mm = float(res[1] / res[2])

    plane = calib.get("fitted_plane")
    if plane and abs(plane["normal"][2]) > 1e-4:
        nx, ny, nz = plane["normal"]
        d = plane["d"]
        z_mm = float(-(nx * x_mm + ny * y_mm + d) / nz)
    else:
        z_mm = float(np.mean([p["robot_position_mm"][2] for p in calib["points"]]))

    return (x_mm, y_mm, z_mm)


def project_robot_to_pixel(
    x_mm: float,
    y_mm: float,
    calib: Union[str, dict] = "matrice_association.json",
) -> Tuple[float, float]:
    """
    Convertit des coordonnées robot (X, Y) en millimètres en pixel image (u, v).

    Args:
        x_mm: X robot en mm.
        y_mm: Y robot en mm.
        calib: Chemin vers le JSON ou dictionnaire chargé.

    Returns:
        (u, v) en pixels sur l'image caméra.
    """
    if isinstance(calib, str):
        calib = load_association(calib)

    H_inv = np.array(calib["homography_robot_mm_to_pixel"], dtype=np.float64)
    vec = np.array([x_mm, y_mm, 1.0], dtype=np.float64)
    res = H_inv @ vec
    if abs(res[2]) < 1e-9:
        raise ValueError("Division par zéro lors de la projection inverse.")
    u = float(res[0] / res[2])
    v = float(res[1] / res[2])
    return (u, v)


# ------------------------------------------------------------------------------
# Point d'entrée en ligne de commande
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Associe interactivement les positions 3D (frames.json) aux coins de l'image (matrice.jpg)."
    )
    parser.add_argument(
        "--image", "-i",
        default="matrice.jpg",
        help="Chemin de l'image de calibration (défaut : matrice.jpg)"
    )
    parser.add_argument(
        "--frames", "-f",
        default="frames.json",
        help="Chemin du fichier frames.json (défaut : frames.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="matrice_association.json",
        help="Fichier JSON de sortie pour l'association et la calibration (défaut : matrice_association.json)"
    )
    parser.add_argument(
        "--annotated", "-a",
        default=None,
        help="Chemin de sauvegarde de l'image annotée de résultat (défaut : <output>_annotated.jpg)"
    )

    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(args.image):
        alt = os.path.join(script_dir, args.image)
        if os.path.exists(alt):
            args.image = alt

    if not os.path.exists(args.frames):
        alt = os.path.join(script_dir, args.frames)
        if os.path.exists(alt):
            args.frames = alt

    app = CornerAssociationCalibrator(
        image_path=args.image,
        frames_path=args.frames,
        output_path=args.output,
        annotated_path=args.annotated,
    )
    app.run()


if __name__ == "__main__":
    main()
