#!/usr/bin/env python3
"""
Préparation du dataset et entraînement de YOLOv8 Nano sur les éprouvettes et le rack.
Utilise le Transfer Learning à partir des poids pré-entraînés 'yolov8n.pt'.
"""

import glob
import os
import random
import shutil
import yaml
import torch
from ultralytics import YOLO


def prepare_dataset(
    images_dir="oak_tools/dataset_raw/all",
    annotations_dir="oak_tools/annotations",
    output_dir="oak_tools/dataset_yolo",
    val_ratio=0.20,
    seed=42,
):
    """Organise les images et annotations dans l'arborescence standard YOLO avec split train/val."""
    random.seed(seed)
    base_dir = os.path.abspath(output_dir)

    train_img_dir = os.path.join(base_dir, "images", "train")
    val_img_dir = os.path.join(base_dir, "images", "val")
    train_lbl_dir = os.path.join(base_dir, "labels", "train")
    val_lbl_dir = os.path.join(base_dir, "labels", "val")

    # Nettoyage et recréation des dossiers
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Récupérer toutes les annotations (support direct et sous-dossiers)
    label_files = glob.glob(os.path.join(annotations_dir, "*.txt"))
    if not label_files:
        label_files = glob.glob(os.path.join(annotations_dir, "**", "*.txt"), recursive=True)
    label_map = {os.path.basename(f).replace(".txt", ".jpg"): f for f in label_files}

    # Séparer par type de caméra (OAK et Wrist) pour un split équilibré
    oak_images = [f for f in label_map.keys() if f.startswith("oak_")]
    wrist_images = [f for f in label_map.keys() if f.startswith("wrist_")]

    random.shuffle(oak_images)
    random.shuffle(wrist_images)

    oak_split = int(len(oak_images) * (1 - val_ratio))
    wrist_split = int(len(wrist_images) * (1 - val_ratio))

    train_files = oak_images[:oak_split] + wrist_images[:wrist_split]
    val_files = oak_images[oak_split:] + wrist_images[wrist_split:]

    print(f"[DATASET] Total images : {len(label_map)}")
    print(f"  -> Entraînement : {len(train_files)} (OAK: {oak_split}, Wrist: {wrist_split})")
    print(f"  -> Validation   : {len(val_files)} (OAK: {len(oak_images)-oak_split}, Wrist: {len(wrist_images)-wrist_split})")

    # Copie des fichiers
    for subset, files, img_target, lbl_target in [
        ("train", train_files, train_img_dir, train_lbl_dir),
        ("val", val_files, val_img_dir, val_lbl_dir),
    ]:
        for img_name in files:
            src_img = os.path.join(images_dir, img_name)
            src_lbl = label_map[img_name]

            shutil.copy2(src_img, os.path.join(img_target, img_name))
            shutil.copy2(src_lbl, os.path.join(lbl_target, img_name.replace(".jpg", ".txt")))

    # Génération du data.yaml
    data_yaml_path = os.path.join(base_dir, "data.yaml")
    data_config = {
        "path": base_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "tube",
            1: "rack",
        },
    }

    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, sort_keys=False)

    print(f"[DATASET] Configuration sauvegardée : {data_yaml_path}")
    return data_yaml_path


def train(data_yaml_path, weights="yolov8n.pt", epochs=50, imgsz=960, batch=8, project="oak_tools/runs", name="tube_detector"):
    """Lance l'entraînement / fine-tuning sur GPU."""
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"\n[TRAIN] Démarrage de l'entraînement sur device={device} avec weights={weights}...")
    if torch.cuda.is_available():
        print(f"[TRAIN] GPU actif : {torch.cuda.get_device_name(0)}")

    model = YOLO(weights)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        val=True,
        lr0=0.005,
        lrf=0.01,
        # Augmentations adaptées à la table
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
    )

    best_weights_src = os.path.join(project, name, "weights", "best.pt")
    best_weights_dst = "oak_tools/weights/best.pt"
    os.makedirs(os.path.dirname(best_weights_dst), exist_ok=True)
    if os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, best_weights_dst)
        print(f"\n[SUCCÈS] Meilleur modèle sauvegardé dans : {best_weights_dst}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement / Fine-tuning YOLOv8")
    parser.add_argument("--weights", default="oak_tools/weights/best_v1_original.pt", help="Poids initiaux pour le fine-tuning")
    parser.add_argument("--data", default="oak_tools/dataset_combined/data.yaml", help="Chemin vers data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=960, help="Taille des images")
    args = parser.parse_args()

    train(args.data, weights=args.weights, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
