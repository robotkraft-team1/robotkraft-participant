#!/usr/bin/env python3
"""
Module de gestion de l'homographie de la table pour RobotKraft.
Permet de transformer la vue caméra (perspective) en vue orthorectifiée (Bird's Eye View)
appliquée à L'ENSEMBLE DE L'IMAGE ou recadrée sur la table, et de convertir des
coordonnées pixels (u, v) en coordonnées métriques table (X, Y) en millimètres.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class TableHomography:
    """Gestionnaire de matrice d'homographie pour la table fixe."""

    def __init__(
        self,
        src_points: Union[np.ndarray, List[List[float]]],
        dst_size_px: Tuple[int, int] = (1920, 1080),
        table_size_mm: Optional[Tuple[float, float]] = None,
        image_size: Optional[Tuple[int, int]] = (1920, 1080),
        zoom: float = 1.0,
        metadata: Optional[dict] = None,
    ):
        """
        Initialise l'homographie.

        Args:
            src_points: 4 points [TL, TR, BR, BL] sur l'image caméra [[x,y], ...].
            dst_size_px: (largeur_px, hauteur_px) de l'image de sortie (par défaut même résolution).
            table_size_mm: (largeur_mm, hauteur_mm) réelles de la table (optionnel).
            image_size: (largeur, hauteur) de l'image caméra d'origine (calibrée).
            zoom: Facteur de zoom / échelle sur la vue globale (défaut: 1.0).
            metadata: Métadonnées additionnelles (date, source, etc.).
        """
        self.src_points = np.array(src_points, dtype=np.float32).reshape((4, 2))
        self.dst_width_px = int(dst_size_px[0])
        self.dst_height_px = int(dst_size_px[1])
        self.table_width_mm = float(table_size_mm[0]) if table_size_mm else None
        self.table_height_mm = float(table_size_mm[1]) if table_size_mm else None
        self.image_size = tuple(image_size) if image_size else (self.dst_width_px, self.dst_height_px)
        self.zoom = float(zoom)
        self.metadata = metadata or {}

        # Calculer les matrices pour l'image entière, le recadrage et les mm
        self._compute_matrices()

    def _compute_matrices(self):
        """Calcule l'ensemble des matrices d'homographie (Image entière, Table seule, Métrique)."""
        src = self.src_points
        img_w, img_h = self.image_size

        # 1. Dimensions naturelles de la zone table
        w_top = np.linalg.norm(src[1] - src[0])
        w_bot = np.linalg.norm(src[2] - src[3])
        h_left = np.linalg.norm(src[3] - src[0])
        h_right = np.linalg.norm(src[2] - src[1])

        # Largeur naturelle sur l'image
        w_table_natural = (w_top + w_bot) / 2.0
        if self.table_width_mm and self.table_height_mm:
            ratio_hw = self.table_height_mm / self.table_width_mm
        else:
            ratio_hw = ((h_left + h_right) / 2.0) / max(1.0, w_table_natural)
        h_table_natural = w_table_natural * ratio_hw

        # Centrage par rapport au barycentre des 4 points dans l'image
        center = src.mean(axis=0)

        # Mise à l'échelle vers la résolution de sortie
        scale_x = self.dst_width_px / float(img_w)
        scale_y = self.dst_height_px / float(img_h)
        base_scale = min(scale_x, scale_y) * self.zoom

        w_t = w_table_natural * base_scale
        h_t = h_table_natural * base_scale
        cx_out = center[0] * scale_x
        cy_out = center[1] * scale_y

        x0 = cx_out - w_t / 2.0
        y0 = cy_out - h_t / 2.0

        # Points de destination pour la vue IMAGE ENTIÈRE (Full Image)
        self.dst_points_full = np.array(
            [
                [x0, y0],              # TL
                [x0 + w_t, y0],        # TR
                [x0 + w_t, y0 + h_t],  # BR
                [x0, y0 + h_t],        # BL
            ],
            dtype=np.float32,
        )

        # 1. Homographie Image Complète (Full Image)
        self.H_full = cv2.getPerspectiveTransform(self.src_points, self.dst_points_full)
        self.H_full_inv = np.linalg.inv(self.H_full)

        # 2. Homographie Recadrée (Crop Table Seule)
        crop_w = max(300, int(round(w_t)))
        crop_h = max(200, int(round(h_t)))
        self.crop_size_px = (crop_w, crop_h)
        self.dst_points_crop = np.array(
            [
                [0.0, 0.0],
                [float(crop_w - 1), 0.0],
                [float(crop_w - 1), float(crop_h - 1)],
                [0.0, float(crop_h - 1)],
            ],
            dtype=np.float32,
        )
        self.H_crop = cv2.getPerspectiveTransform(self.src_points, self.dst_points_crop)
        self.H_crop_inv = np.linalg.inv(self.H_crop)

        # 3. Homographie Métrique (Millimètres réels sur le plan de la table)
        if self.table_width_mm and self.table_height_mm:
            self.dst_points_mm = np.array(
                [
                    [0.0, 0.0],
                    [self.table_width_mm, 0.0],
                    [self.table_width_mm, self.table_height_mm],
                    [0.0, self.table_height_mm],
                ],
                dtype=np.float32,
            )
            self.H_mm = cv2.getPerspectiveTransform(self.src_points, self.dst_points_mm)
            self.H_mm_inv = np.linalg.inv(self.H_mm)
            self.px_per_mm_x = w_t / self.table_width_mm
            self.px_per_mm_y = h_t / self.table_height_mm
        else:
            self.dst_points_mm = None
            self.H_mm = None
            self.H_mm_inv = None
            self.px_per_mm_x = None
            self.px_per_mm_y = None

        # Alias par défaut : H_px = H_full
        self.H_px = self.H_full
        self.H_px_inv = self.H_full_inv
        self.dst_points_px = self.dst_points_full

    def _get_scaled_h(
        self,
        current_img_shape: Tuple[int, int, ...],
        mode: str = "full",
    ) -> np.ndarray:
        """Ajuste l'homographie si la résolution de l'image caméra courante diffère de la calibration."""
        H = self.H_full if mode == "full" else self.H_crop
        if self.image_size is None:
            return H

        curr_h, curr_w = current_img_shape[:2]
        orig_w, orig_h = self.image_size

        if curr_w == orig_w and curr_h == orig_h:
            return H

        sx = curr_w / float(orig_w)
        sy = curr_h / float(orig_h)

        scale_mat_inv = np.array(
            [
                [1.0 / sx, 0.0, 0.0],
                [0.0, 1.0 / sy, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return H @ scale_mat_inv

    def warp_image(
        self,
        image: np.ndarray,
        mode: str = "full",
        output_size: Optional[Tuple[int, int]] = None,
        flags: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: Tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Applique la transformation d'homographie.

        Args:
            image: Image caméra source.
            mode: "full" pour transformer l'ENSEMBLE de l'image (défaut),
                  "crop" pour recadrer uniquement sur la table.
            output_size: (w, h) de l'image résultante.
            flags: Interpolation cv2 (ex: cv2.INTER_LINEAR, cv2.INTER_CUBIC).
            border_mode: Traitement des bords hors champ.
            border_value: Couleur de fond pour les zones hors champ.

        Returns:
            Image redressée par homographie.
        """
        H = self._get_scaled_h(image.shape, mode=mode)
        if output_size:
            w, h = output_size
        elif mode == "full":
            w, h = self.dst_width_px, self.dst_height_px
        else:
            w, h = self.crop_size_px

        return cv2.warpPerspective(
            image,
            H,
            (w, h),
            flags=flags,
            borderMode=border_mode,
            borderValue=border_value,
        )

    def pixel_to_table(
        self,
        u: float,
        v: float,
        in_mm: bool = True,
        mode: str = "full",
        current_img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """
        Convertit un pixel caméra (u, v) en coordonnées de la table.

        Args:
            u: X dans l'image caméra.
            v: Y dans l'image caméra.
            in_mm: Si True, retourne en millimètres réels (X_mm, Y_mm).
                   Si False, retourne en coordonnées pixel de l'image redressée.
            mode: "full" (pixel dans la vue globale) ou "crop" (pixel table recadrée).
            current_img_size: Taille actuelle si différente de la calibration.

        Returns:
            (X, Y)
        """
        if current_img_size and self.image_size and current_img_size != self.image_size:
            sx = current_img_size[0] / float(self.image_size[0])
            sy = current_img_size[1] / float(self.image_size[1])
            u = u / sx
            v = v / sy

        if in_mm and self.H_mm is not None:
            H = self.H_mm
        elif mode == "full":
            H = self.H_full
        else:
            H = self.H_crop

        vec = np.array([u, v, 1.0], dtype=np.float64)
        res = H @ vec
        if abs(res[2]) < 1e-9:
            return float("nan"), float("nan")
        return float(res[0] / res[2]), float(res[1] / res[2])

    def table_to_pixel(
        self,
        x: float,
        y: float,
        from_mm: bool = True,
        mode: str = "full",
        target_img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[float, float]:
        """
        Convertit une coordonnée de la table (mm ou pixel vue redressée) vers le pixel caméra (u, v).
        """
        if from_mm and self.H_mm_inv is not None:
            H_inv = self.H_mm_inv
        elif mode == "full":
            H_inv = self.H_full_inv
        else:
            H_inv = self.H_crop_inv

        vec = np.array([x, y, 1.0], dtype=np.float64)
        res = H_inv @ vec
        if abs(res[2]) < 1e-9:
            return float("nan"), float("nan")
        u = float(res[0] / res[2])
        v = float(res[1] / res[2])

        if target_img_size and self.image_size and target_img_size != self.image_size:
            sx = target_img_size[0] / float(self.image_size[0])
            sy = target_img_size[1] / float(self.image_size[1])
            u *= sx
            v *= sy

        return u, v

    def points_to_table(
        self,
        points: Union[np.ndarray, List[Tuple[float, float]]],
        in_mm: bool = True,
        mode: str = "full",
        current_img_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Transforme un lot de points Nx2 de l'image caméra vers le plan de la table."""
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        if current_img_size and self.image_size and current_img_size != self.image_size:
            sx = current_img_size[0] / float(self.image_size[0])
            sy = current_img_size[1] / float(self.image_size[1])
            pts[:, 0, 0] /= sx
            pts[:, 0, 1] /= sy

        if in_mm and self.H_mm is not None:
            H = self.H_mm
        elif mode == "full":
            H = self.H_full
        else:
            H = self.H_crop

        transformed = cv2.perspectiveTransform(pts, H)
        return transformed.reshape((-1, 2))

    def points_to_pixel(
        self,
        points: Union[np.ndarray, List[Tuple[float, float]]],
        from_mm: bool = True,
        mode: str = "full",
        target_img_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Transforme un lot de points Nx2 de la table vers l'image caméra."""
        pts = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        if from_mm and self.H_mm_inv is not None:
            H_inv = self.H_mm_inv
        elif mode == "full":
            H_inv = self.H_full_inv
        else:
            H_inv = self.H_crop_inv

        transformed = cv2.perspectiveTransform(pts, H_inv)
        res = transformed.reshape((-1, 2))

        if target_img_size and self.image_size and target_img_size != self.image_size:
            sx = target_img_size[0] / float(self.image_size[0])
            sy = target_img_size[1] / float(self.image_size[1])
            res[:, 0] *= sx
            res[:, 1] *= sy

        return res

    def is_inside_table(
        self,
        u: float,
        v: float,
        current_img_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Vérifie si un pixel caméra (u, v) est à l'intérieur du polygone des 4 points de la table."""
        src_pts = self.src_points.copy()
        if current_img_size and self.image_size and current_img_size != self.image_size:
            sx = current_img_size[0] / float(self.image_size[0])
            sy = current_img_size[1] / float(self.image_size[1])
            src_pts[:, 0] *= sx
            src_pts[:, 1] *= sy

        poly = src_pts.astype(np.int32)
        dist = cv2.pointPolygonTest(poly, (float(u), float(v)), False)
        return dist >= 0

    def draw_table_boundary(
        self,
        image: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill_alpha: float = 0.15,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """Dessine le quadrilatère de la table calibrée sur l'image caméra."""
        out = image.copy()
        h, w = out.shape[:2]

        pts = self.src_points.copy()
        if self.image_size and (w, h) != self.image_size:
            sx = w / float(self.image_size[0])
            sy = h / float(self.image_size[1])
            pts[:, 0] *= sx
            pts[:, 1] *= sy

        pts_int = pts.astype(np.int32)

        if fill_alpha > 0.0:
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts_int], color)
            cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0, out)

        cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        labels = ["P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)"]
        corner_colors = [
            (255, 255, 0),  # Cyan
            (0, 255, 0),    # Vert
            (0, 215, 255),  # Jaune
            (0, 140, 255),  # Orange
        ]
        for i, (pt, label, col) in enumerate(zip(pts_int, labels, corner_colors)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(out, (x, y), 6, col, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 8, (0, 0, 0), 1, cv2.LINE_AA)
            if draw_labels:
                offset_y = -10 if i in [0, 1] else 20
                cv2.putText(
                    out,
                    label,
                    (x + 10, y + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    label,
                    (x + 10, y + offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    col,
                    1,
                    cv2.LINE_AA,
                )

        return out

    def draw_grid_on_image(
        self,
        image: np.ndarray,
        step_mm: float = 100.0,
        color: Tuple[int, int, int] = (255, 200, 0),
        thickness: int = 1,
    ) -> np.ndarray:
        """Projette la grille métrique virtuelle sur l'image caméra."""
        if self.H_mm_inv is None or not self.table_width_mm or not self.table_height_mm:
            return image

        out = image.copy()
        curr_h, curr_w = out.shape[:2]
        target_size = (curr_w, curr_h)

        xs = np.arange(0, self.table_width_mm + 0.1, step_mm)
        for x in xs:
            ys = np.linspace(0, self.table_height_mm, 20)
            pts_table = np.column_stack([np.full_like(ys, x), ys])
            pts_cam = self.points_to_pixel(pts_table, from_mm=True, target_img_size=target_size)
            pts_cam = pts_cam.astype(np.int32)
            for j in range(len(pts_cam) - 1):
                cv2.line(out, tuple(pts_cam[j]), tuple(pts_cam[j + 1]), color, thickness, cv2.LINE_AA)

        ys = np.arange(0, self.table_height_mm + 0.1, step_mm)
        for y in ys:
            xs_line = np.linspace(0, self.table_width_mm, 20)
            pts_table = np.column_stack([xs_line, np.full_like(xs_line, y)])
            pts_cam = self.points_to_pixel(pts_table, from_mm=True, target_img_size=target_size)
            pts_cam = pts_cam.astype(np.int32)
            for j in range(len(pts_cam) - 1):
                cv2.line(out, tuple(pts_cam[j]), tuple(pts_cam[j + 1]), color, thickness, cv2.LINE_AA)

        return out

    def draw_table_on_warped(
        self,
        warped_image: np.ndarray,
        mode: str = "full",
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Dessine l'emprise rectangulaire de la table sur la vue redressée."""
        out = warped_image.copy()
        pts = self.dst_points_full if mode == "full" else self.dst_points_crop
        pts_int = pts.astype(np.int32)
        cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        return out

    def to_dict(self) -> dict:
        """Sérialise en dictionnaire JSON."""
        return {
            "metadata": {
                **self.metadata,
                "created_at": self.metadata.get("created_at", datetime.now().isoformat()),
                "version": "2.0",
                "warp_default": "full",
            },
            "image_size": [int(x) for x in self.image_size] if self.image_size else None,
            "table_size_mm": [float(x) for x in (self.table_width_mm, self.table_height_mm)] if self.table_width_mm else None,
            "dst_size_px": [int(self.dst_width_px), int(self.dst_height_px)],
            "crop_size_px": [int(x) for x in self.crop_size_px],
            "zoom": float(self.zoom),
            "src_points": self.src_points.tolist(),
            "dst_points_full": self.dst_points_full.tolist(),
            "dst_points_crop": self.dst_points_crop.tolist(),
            "dst_points_mm": self.dst_points_mm.tolist() if self.dst_points_mm is not None else None,
            "H_full": self.H_full.tolist(),
            "H_full_inv": self.H_full_inv.tolist(),
            "H_crop": self.H_crop.tolist(),
            "H_crop_inv": self.H_crop_inv.tolist(),
            "H_mm": self.H_mm.tolist() if self.H_mm is not None else None,
            "H_mm_inv": self.H_mm_inv.tolist() if self.H_mm_inv is not None else None,
            "px_per_mm": [float(self.px_per_mm_x), float(self.px_per_mm_y)] if self.px_per_mm_x is not None else None,
            # Compatibilité ascendante v1.0
            "H_px": self.H_full.tolist(),
            "H_px_inv": self.H_full_inv.tolist(),
            "dst_points_px": self.dst_points_full.tolist(),
        }

    def save(self, filepath: str = "table_homography.json"):
        """Sauvegarde dans un fichier JSON."""
        data = self.to_dict()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[TableHomography] Configuration sauvegardée dans : {os.path.abspath(filepath)}")

    @classmethod
    def from_dict(cls, data: dict) -> "TableHomography":
        """Instancie depuis un dictionnaire JSON."""
        src_points = data["src_points"]
        dst_size_px = tuple(data.get("dst_size_px", (1920, 1080)))
        table_size_mm = tuple(data["table_size_mm"]) if data.get("table_size_mm") else None
        image_size = tuple(data["image_size"]) if data.get("image_size") else None
        zoom = float(data.get("zoom", 1.0))
        metadata = data.get("metadata", {})
        return cls(
            src_points=src_points,
            dst_size_px=dst_size_px,
            table_size_mm=table_size_mm,
            image_size=image_size,
            zoom=zoom,
            metadata=metadata,
        )

    @classmethod
    def load(cls, filepath: str = "table_homography.json") -> "TableHomography":
        """Charge une configuration JSON."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier de calibration introuvable : {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
