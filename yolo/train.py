#!/usr/bin/env python3
"""Entraîne YOLOv8 sur le dataset produit par labelme2yolo.py.

Usage :
    python yolo/train.py                                  # yolov8n, 100 epochs, 640 px
    python yolo/train.py --model yolov8s.pt --epochs 200 --imgsz 960
    python yolo/train.py --resume                         # reprend le dernier run interrompu

Le modèle de départ suit la tâche du dataset (ligne "--task" de data.yaml) : yolov8n.pt en détection,
yolov8n-seg.pt en segmentation. Poids : yolo/runs/<task>/<name>/weights/best.pt

Augmentations (globe_augment.py, à visualiser avec preview_augment.py) : rotation légère recadrée sans bord
gris, zoom avant seulement, luminosité/saturation/teinte peu modifiées. Ni mosaïque, ni flip (une carte en
miroir n'est plus la même carte), ni recul, ni mixup.
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

import globe_augment

HERE = Path(__file__).resolve().parent

# Partagé avec preview_augment.py, qui affiche les images telles que l'entraînement les voit.
# degrees et scale sont interprétés par globe_augment.RotateZoomCrop : rotation max (°), zoom avant max (+30 %).
AUGMENT = dict(
    degrees=5.0,
    scale=0.3,
    hsv_h=0.005,
    hsv_s=0.2,
    hsv_v=0.15,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
    close_mosaic=0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    translate=0.0,
    shear=0.0,
    perspective=0.0,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=HERE / "data/globe/data.yaml")
    p.add_argument("--model", help="Poids de départ (défaut : yolov8n.pt ou yolov8n-seg.pt selon la tâche)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="-1 = auto selon la VRAM")
    p.add_argument("--device", default=None, help="0, cpu... (défaut : GPU si dispo)")
    p.add_argument("--name", default="globe")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def dataset_task(data_yaml: Path) -> str:
    first = data_yaml.read_text().splitlines()[0]
    return "segment" if "--task segment" in first else "detect"


def main():
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"{args.data} introuvable : lancer d'abord yolo/labelme2yolo.py (make yolo-convert)")
    task = dataset_task(args.data)
    args.data = args.data.resolve()
    if args.model and Path(args.model).exists():
        args.model = str(Path(args.model).resolve())
    os.chdir(HERE)  # les poids pré-entraînés (yolov8n.pt...) sont téléchargés dans yolo/, pas à la racine
    project = HERE / "runs" / task

    if args.resume:
        last = max(project.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime, default=None)
        if last is None:
            raise SystemExit(f"Aucun last.pt dans {project}")
        globe_augment.install()
        YOLO(last).train(resume=True)
        return

    globe_augment.install()
    model = YOLO(args.model or ("yolov8n-seg.pt" if task == "segment" else "yolov8n.pt"))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(project),
        name=args.name,
        exist_ok=False,
        seed=0,
        plots=True,
        **AUGMENT,
    )
    print(f"\nmAP50 = {results.box.map50:.3f}   mAP50-95 = {results.box.map:.3f}")
    print(f"Poids : {Path(model.trainer.best)}")


if __name__ == "__main__":
    main()
