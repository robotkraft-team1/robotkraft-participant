#!/usr/bin/env python3
"""Convertit les annotations labelme (un .json par image) en dataset YOLO prêt pour l'entraînement.

Entrée : les JSON labelme (par défaut à côté des images, comme labelme les enregistre) :
    yolo/datasets/globe/images/top_20261003_181800_0036.jpg
    yolo/datasets/globe/images/top_20261003_181800_0036.json
Sortie (OUT, défaut yolo/data/globe) :
    images/{train,val}/*.jpg   labels/{train,val}/*.txt   data.yaml   classes.txt   [preview/*.jpg]

Le split train/val se fait par capture (<session>_<nnnn>) : l'image top et l'image poignet d'un même
instant tombent toujours dans le même split, sinon la validation serait optimiste.

Formes labelme gérées : point, rectangle, polygon, circle. En --task detect chaque forme devient sa boîte
englobante ; en --task segment un polygone (rectangle -> 4 coins, cercle -> 32 côtés). Un point (repère lat/lon
cliqué sur le globe) devient une boîte carrée centrée dessus, de côté --point-box (fraction de la largeur de
l'image si <= 1, sinon en pixels) : le centre de la boîte prédite redonne la position du repère.

Usage :
    python yolo/labelme2yolo.py                             # détection, toutes caméras
    python yolo/labelme2yolo.py --cameras top --val 0.25
    python yolo/labelme2yolo.py --point-box 0.03 --preview      # boîtes plus petites autour des points
    python yolo/labelme2yolo.py --classes amerique_sud,afrique,europe   # ordre des classes imposé
"""

import argparse
import json
import math
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

CAPTURE_RE = re.compile(r"^(?P<cam>[a-z]+)_(?P<capture>\d{8}_\d{6}_\d+)$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path("yolo/datasets/globe/images"), help="Dossier des JSON labelme (récursif)")
    p.add_argument("--images", type=Path, help="Dossier des images si différent de --src")
    p.add_argument("--out", type=Path, default=Path("yolo/data/globe"))
    p.add_argument("--task", choices=["detect", "segment"], default="detect")
    p.add_argument("--cameras", default="top,wrist", help="Préfixes de caméra à garder (défaut top,wrist)")
    p.add_argument("--classes", help="Liste ordonnée, séparée par des virgules (défaut : labels trouvés, triés)")
    p.add_argument("--val", type=float, default=0.2, help="Fraction de captures en validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point-box", type=float, default=0.04,
                   help="Côté de la boîte autour d'un point : fraction de la largeur si <= 1, sinon pixels (0 = ignorer les points)")
    p.add_argument("--include-empty", action="store_true", help="Garde les images sans JSON comme exemples négatifs")
    p.add_argument("--preview", action="store_true", help="Dessine les labels convertis dans OUT/preview pour vérification")
    return p.parse_args()


def capture_key(stem: str) -> tuple[str, str]:
    """('top', '20261003_181800_0036') ; sinon (stem, stem) pour les noms hors convention."""
    m = CAPTURE_RE.match(stem)
    return (m["cam"], m["capture"]) if m else ("", stem)


def find_image(json_path: Path, ann: dict, images_dir: Path | None) -> Path | None:
    candidates = []
    if ann.get("imagePath"):
        candidates.append(json_path.parent / ann["imagePath"])
    for d in filter(None, (images_dir, json_path.parent)):
        candidates += [d / (json_path.stem + ext) for ext in IMAGE_EXTS]
    return next((c.resolve() for c in candidates if c.is_file()), None)


def shape_polygon(shape: dict, point_box: float, img_w: int) -> list[tuple[float, float]] | None:
    """Points (en pixels) décrivant la forme, ou None si elle n'est pas convertible."""
    pts = [tuple(map(float, pt)) for pt in shape.get("points", [])]
    kind = shape.get("shape_type") or "polygon"
    if kind == "rectangle" and len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if kind == "circle" and len(pts) == 2:
        (cx, cy), (ex, ey) = pts
        r = math.hypot(ex - cx, ey - cy)
        return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (2 * math.pi * i / 32 for i in range(32))]
    if kind == "point" and len(pts) == 1 and point_box > 0:
        (x, y), h = pts[0], (point_box * img_w if point_box <= 1 else point_box) / 2
        return [(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]
    if kind == "polygon" and len(pts) >= 3:
        return pts
    return None


def clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def yolo_line(cls: int, poly: list[tuple[float, float]], w: int, h: int, task: str) -> str | None:
    norm = [(clamp01(x / w), clamp01(y / h)) for x, y in poly]
    if task == "segment":
        return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
    xs, ys = [x for x, _ in norm], [y for _, y in norm]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    if bw <= 0 or bh <= 0:
        return None
    return f"{cls} {min(xs) + bw / 2:.6f} {min(ys) + bh / 2:.6f} {bw:.6f} {bh:.6f}"


def load_samples(args) -> tuple[list[dict], list[str]]:
    cameras = {c.strip() for c in args.cameras.split(",") if c.strip()}
    samples, warnings = [], []
    json_files = sorted(args.src.rglob("*.json"))
    for jp in json_files:
        try:
            ann = json.loads(jp.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "shapes" not in ann:  # pas un fichier labelme (ex. camera_config.json)
            continue
        cam, capture = capture_key(jp.stem)
        if cam and cam not in cameras:
            continue
        img = find_image(jp, ann, args.images)
        if img is None:
            warnings.append(f"{jp.name} : image introuvable")
            continue
        with Image.open(img) as im:
            w, h = im.size
        if (ann.get("imageWidth"), ann.get("imageHeight")) not in ((w, h), (None, None)):
            warnings.append(f"{jp.name} : taille JSON {ann['imageWidth']}x{ann['imageHeight']} != image {w}x{h}, image utilisée")
        shapes = []
        for s in ann["shapes"]:
            poly = shape_polygon(s, args.point_box, w)
            if poly is None:
                warnings.append(f"{jp.name} : forme '{s.get('label')}' ({s.get('shape_type')}) ignorée")
                continue
            shapes.append((s["label"].strip(), poly))
        samples.append({"image": img, "w": w, "h": h, "shapes": shapes, "capture": capture})

    if args.include_empty:
        done = {s["image"].stem for s in samples}
        img_dir = args.images or args.src
        for img in sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            cam, capture = capture_key(img.stem)
            if img.stem in done or (cam and cam not in cameras):
                continue
            with Image.open(img) as im:
                w, h = im.size
            samples.append({"image": img.resolve(), "w": w, "h": h, "shapes": [], "capture": capture})
    return samples, warnings


def split_captures(samples: list[dict], val: float, seed: int) -> dict[str, str]:
    captures = sorted({s["capture"] for s in samples})
    random.Random(seed).shuffle(captures)
    n_val = round(len(captures) * val) if len(captures) > 1 else 0
    if val > 0 and len(captures) > 1:
        n_val = max(n_val, 1)
    return {c: ("val" if i < n_val else "train") for i, c in enumerate(captures)}


def draw_preview(sample: dict, lines: list[str], names: list[str], dest: Path, task: str):
    with Image.open(sample["image"]) as im:
        im = im.convert("RGB")
    d, w, h = ImageDraw.Draw(im), sample["w"], sample["h"]
    for line in lines:
        cls, *v = line.split()
        v = list(map(float, v))
        if task == "segment":
            pts = [(v[i] * w, v[i + 1] * h) for i in range(0, len(v), 2)]
            d.polygon(pts, outline=(255, 0, 0), width=3)
            anchor = pts[0]
        else:
            cx, cy, bw, bh = v
            anchor = ((cx - bw / 2) * w, (cy - bh / 2) * h)
            d.rectangle([anchor, ((cx + bw / 2) * w, (cy + bh / 2) * h)], outline=(255, 0, 0), width=3)
        d.text((anchor[0] + 4, anchor[1] + 4), names[int(cls)], fill=(255, 255, 0))
    im.save(dest, quality=85)


def main():
    args = parse_args()
    if not args.src.is_dir():
        sys.exit(f"Dossier introuvable : {args.src}")

    samples, warnings = load_samples(args)
    for w in warnings:
        print("  ! " + w)
    annotated = [s for s in samples if s["shapes"]]
    if not annotated:
        sys.exit(f"Aucune annotation labelme trouvée dans {args.src} (JSON avec une clé 'shapes').")

    found = sorted({label for s in samples for label, _ in s["shapes"]})
    names = [c.strip() for c in args.classes.split(",")] if args.classes else found
    unknown = set(found) - set(names)
    if unknown:
        sys.exit(f"Labels absents de --classes : {', '.join(sorted(unknown))}")
    cls_id = {n: i for i, n in enumerate(names)}

    if args.out.exists():
        shutil.rmtree(args.out)
    split_of = split_captures(samples, args.val, args.seed)
    counts = {"train": Counter(), "val": Counter()}
    n_images = Counter()
    for s in samples:
        split = split_of[s["capture"]]
        lines = [l for label, poly in s["shapes"] if (l := yolo_line(cls_id[label], poly, s["w"], s["h"], args.task))]
        img_dst = args.out / "images" / split / s["image"].name
        lbl_dst = args.out / "labels" / split / (s["image"].stem + ".txt")
        img_dst.parent.mkdir(parents=True, exist_ok=True)
        lbl_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s["image"], img_dst)
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images[split] += 1
        counts[split].update(names[int(l.split()[0])] for l in lines)
        if args.preview and lines:
            (args.out / "preview").mkdir(exist_ok=True)
            draw_preview(s, lines, names, args.out / "preview" / s["image"].name, args.task)

    out = args.out.resolve()
    (args.out / "classes.txt").write_text("\n".join(names) + "\n")
    (args.out / "data.yaml").write_text(
        f"# Généré par yolo/labelme2yolo.py (--task {args.task})\n"
        f"path: {out}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {json.dumps(n, ensure_ascii=False)}\n" for i, n in enumerate(names))
    )

    print(f"\nDataset YOLO ({args.task}) -> {out}")
    print(f"  images : train {n_images['train']}, val {n_images['val']}  ({len(annotated)} annotées / {len(samples)})")
    print(f"  {'classe':<24}{'train':>7}{'val':>7}")
    for n in names:
        print(f"  {n:<24}{counts['train'][n]:>7}{counts['val'][n]:>7}")
    missing_val = [n for n in names if counts["val"][n] == 0]
    if missing_val and n_images["val"]:
        print(f"  ! classes absentes de la validation : {', '.join(missing_val)} (essayer un autre --seed ou --val)")


if __name__ == "__main__":
    main()
