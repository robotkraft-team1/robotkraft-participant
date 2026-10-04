"""Augmentation adaptée au globe, qui remplace le pipeline par défaut d'ultralytics (v8_transforms).

Le pipeline par défaut (mosaïque, mixup, recul, translation, rotation avec bords gris...) produit des
scènes irréalistes pour ce problème. Ici, sur l'image d'origine (avant letterbox), on applique seulement :
  1. RotateZoomCrop : rotation de ±degrees puis zoom avant juste suffisant pour que l'image remplisse tout le
     cadre (aucune zone grise), plus un zoom supplémentaire aléatoire de 1 à 1+scale et un recadrage aléatoire
     dans la marge disponible. Jamais de recul (zoom < 1).
  2. LetterBox au format carré imgsz (le même que la validation et l'inférence).
  3. RandomHSV avec des gains faibles (hsv_h, hsv_s, hsv_v).
Les hyperparamètres ultralytics `degrees` et `scale` sont donc réinterprétés (rotation max, zoom avant max),
et `mosaic`, `translate`, `shear`, `perspective`, `mixup`, `cutmix`, `copy_paste` n'ont plus d'effet.

install() doit être appelé avant la construction du dataset (train.py, preview_augment.py).
"""

import math
import random

import cv2
import numpy as np
from ultralytics.data import dataset as yolo_dataset
from ultralytics.data.augment import Compose, LetterBox, RandomFlip, RandomHSV
from ultralytics.utils.instance import Instances


class RotateZoomCrop:
    def __init__(self, degrees: float = 0.0, zoom: float = 0.0, min_visible: float = 0.5):
        self.degrees = degrees
        self.zoom = zoom
        self.min_visible = min_visible  # fraction minimale de la boîte restant dans le cadre pour la garder

    @staticmethod
    def _covers(m: np.ndarray, w: int, h: int) -> bool:
        """Vrai si tout le cadre de sortie provient de l'intérieur de l'image source (pas de bord rajouté)."""
        inv = cv2.invertAffineTransform(m)
        corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64)
        src = corners @ inv.T
        eps = 1e-3
        return bool((src[:, 0] >= -eps).all() and (src[:, 0] <= w + eps).all()
                    and (src[:, 1] >= -eps).all() and (src[:, 1] <= h + eps).all())

    def _matrix(self, w: int, h: int) -> np.ndarray:
        angle = random.uniform(-self.degrees, self.degrees)
        t = math.radians(abs(angle))
        # zoom minimal pour qu'un cadre w x h tourné de t reste couvert par l'image tournée
        s_min = math.cos(t) + max(w, h) / min(w, h) * math.sin(t)
        s = s_min * random.uniform(1.0, 1.0 + self.zoom)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
        for _ in range(20):  # recadrage aléatoire dans la marge laissée par le zoom
            shifted = m.copy()
            shifted[:, 2] += (random.uniform(-1, 1) * (s - 1) * w / 2, random.uniform(-1, 1) * (s - 1) * h / 2)
            if self._covers(shifted, w, h):
                return shifted
        return m

    def __call__(self, labels: dict) -> dict:
        img = labels["img"]
        h, w = img.shape[:2]
        inst: Instances = labels.pop("instances")
        inst.convert_bbox("xyxy")
        inst.denormalize(w, h)
        if self.degrees == 0 and self.zoom == 0:
            labels["instances"] = inst
            return labels

        m = self._matrix(w, h)
        # BORDER_REFLECT n'est qu'une sécurité : _matrix garantit que le cadre est entièrement couvert
        labels["img"] = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        boxes = inst.bboxes
        n = len(boxes)
        corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(n * 4, 2)
        corners = (np.c_[corners, np.ones(n * 4)] @ m.T).reshape(n, 4, 2)
        new = np.c_[corners.min(1), corners.max(1)]
        clipped = new.copy()
        clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, w)
        clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, h)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = area(clipped) >= self.min_visible * np.maximum(area(new), 1e-6)

        segments = inst.segments
        if len(segments):
            segments = np.c_[segments.reshape(-1, 2), np.ones(segments.shape[0] * segments.shape[1])] @ m.T
            segments = segments.reshape(inst.segments.shape)
            segments[..., 0] = segments[..., 0].clip(0, w)
            segments[..., 1] = segments[..., 1].clip(0, h)
        out = Instances(clipped.astype(np.float32), segments, inst.keypoints, bbox_format="xyxy", normalized=False)
        labels["instances"] = out[keep]
        labels["cls"] = labels["cls"][keep]
        return labels


def globe_transforms(dataset, imgsz: int, hyp) -> Compose:
    return Compose([
        RotateZoomCrop(degrees=hyp.degrees, zoom=hyp.scale),
        LetterBox(new_shape=(imgsz, imgsz)),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction="vertical", p=hyp.flipud),
        RandomFlip(direction="horizontal", p=hyp.fliplr),
    ])


def install():
    """Fait utiliser globe_transforms par YOLODataset.build_transforms (entraînement uniquement)."""
    yolo_dataset.v8_transforms = globe_transforms
