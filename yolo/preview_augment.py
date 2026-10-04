#!/usr/bin/env python3
"""Affiche les images augmentées avec leurs boîtes, exactement comme l'entraînement les voit.

Utilise le pipeline de données d'ultralytics avec l'augmentation de globe_augment.py et les mêmes paramètres
que train.py (AUGMENT), pour vérifier que les boîtes suivent bien les repères et que l'augmentation reste réaliste.

Fenêtre : ESPACE/N = nouvelle planche, S = sauvegarder, Q/Échap = quitter.
Sans écran (ou --no-show) : sauvegarde --pages planches dans --save.

Usage :
    python yolo/preview_augment.py                         # planches 4x4 du split train augmenté
    python yolo/preview_augment.py --split val             # sans augmentation (letterbox seul)
    python yolo/preview_augment.py --set degrees=10 --set hsv_v=0.3   # tester d'autres réglages
    python yolo/preview_augment.py --no-show --pages 5     # 5 planches dans yolo/runs/augment_preview/
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.plotting import colors

import globe_augment
from train import AUGMENT, HERE, dataset_task


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "data/globe/data.yaml")
    p.add_argument("--split", choices=["train", "val"], default="train", help="val = sans augmentation")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--n", type=int, default=16, help="Images par planche")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--set", action="append", default=[], metavar="CLE=VAL", help="Surcharge d'un hyperparamètre")
    p.add_argument("--save", type=Path, default=HERE / "runs/augment_preview")
    p.add_argument("--no-show", action="store_true", help="Pas de fenêtre, sauvegarde seulement")
    p.add_argument("--pages", type=int, default=1, help="Planches sauvegardées avec --no-show")
    p.add_argument("--seed", type=int)
    return p.parse_args()


def parse_value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def build_dataset(args):
    globe_augment.install()
    data = check_det_dataset(str(args.data))
    overrides = dict(task=dataset_task(args.data), imgsz=args.imgsz, **AUGMENT)
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k.strip()] = parse_value(v.strip())
    cfg = get_cfg(overrides=overrides)
    mode = "train" if args.split == "train" else "val"
    return build_yolo_dataset(cfg, data[args.split], batch=1, data=data, mode=mode), data["names"]


def draw(sample: dict, names: dict) -> np.ndarray:
    img = sample["img"].numpy().transpose(1, 2, 0)[..., ::-1].copy()  # tenseur CHW RGB -> HWC BGR
    h, w = img.shape[:2]
    for c, (cx, cy, bw, bh) in zip(sample["cls"].flatten().int().tolist(), sample["bboxes"].tolist()):
        x1, y1, x2, y2 = (int(v) for v in ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        color = colors(c, bgr=True)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.drawMarker(img, ((x1 + x2) // 2, (y1 + y2) // 2), color, cv2.MARKER_CROSS, 8, 1)
        label = names[c]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(img, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        text_color = (0, 0, 0) if sum(color) > 450 else (255, 255, 255)  # lisible sur fond clair
        cv2.putText(img, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    src = Path(sample["im_file"]).stem
    cv2.putText(img, f"{src}  ({len(sample['cls'])} boites)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, f"{src}  ({len(sample['cls'])} boites)", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def make_page(dataset, names, n: int, cols: int) -> np.ndarray:
    tiles = [draw(dataset[random.randrange(len(dataset))], names) for _ in range(n)]
    h, w = tiles[0].shape[:2]
    rows = -(-n // cols)
    page = np.full((rows * h, cols * w, 3), 114, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        page[r * h : r * h + t.shape[0], c * w : c * w + t.shape[1]] = t
    return page


def save(page: np.ndarray, out_dir: Path, split: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    i = len(list(out_dir.glob(f"{split}_*.jpg")))
    path = out_dir / f"{split}_{i:03d}.jpg"
    cv2.imwrite(str(path), page)
    print(f"  -> {path}")
    return path


def window_closed(win: str) -> bool:
    try:
        return cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:  # backend Qt : la fenêtre peut ne pas être encore prête au premier tour
        return False


def main():
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"{args.data} introuvable : lancer d'abord make yolo-convert")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    dataset, names = build_dataset(args)
    show = not args.no_show and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if not show:
        for _ in range(args.pages):
            save(make_page(dataset, names, args.n, args.cols), args.save, args.split)
        return

    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # le Qt embarqué d'opencv-python n'a que le plugin X11
    win = "augmentations - ESPACE: suivant  S: sauver  Q: quitter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    page = make_page(dataset, names, args.n, args.cols)
    scale = min(1.0, 1600 / page.shape[1], 950 / page.shape[0])
    cv2.resizeWindow(win, int(page.shape[1] * scale), int(page.shape[0] * scale))
    while True:
        cv2.imshow(win, page)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27) or window_closed(win):
            break
        if key == ord("s"):
            save(page, args.save, args.split)
        elif key in (ord(" "), ord("n")):
            page = make_page(dataset, names, args.n, args.cols)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
