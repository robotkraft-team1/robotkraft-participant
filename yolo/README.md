# YOLOv8 — repères lat/lon sur le globe

Détecte sur les images (caméra OAK-D « top » et caméra poignet) les repères annotés au point dans labelme.
Chaque classe porte le nom de sa coordonnée (ex. `48°28'24.75"N 4°58'49.94"W`). Le centre de la boîte
prédite donne la position du repère dans l'image, c'est ce qui servira à viser le point pour la photo.

Tourne sur l'hôte (pas dans Docker), dans un venv dédié `yolo/.venv`, sur le GPU NVIDIA s'il est disponible.
Caméras : OAK-D Lite (top) et caméra poignet `/dev/video2` (surchargeable avec `WRIST_DEVICE=`).

```
make yolo-setup                      # une fois : venv + ultralytics
make yolo-convert PREVIEW=1          # labelme -> yolo/data/globe (+ preview/ pour vérifier les boîtes)
make yolo-preview-aug               # planches d'images augmentées + boîtes (ESPACE suivant, S sauver, Q quitter)
make yolo-train                      # yolov8n, 100 epochs, 640 px -> yolo/runs/detect/globe/weights/best.pt
make yolo-live                       # temps réel sur les caméras (Q quitter, S snapshot, +/- confiance)
make yolo-teleop                     # idem + téléopération leader -> follower pour viser avec le bras
make yolo-predict                    # inférence sur le split val -> yolo/runs/predict/
```

| Dossier | Contenu |
|---|---|
| `datasets/globe/images/` | images + JSON labelme (source, une annotation par image) |
| `data/globe/` | dataset YOLO généré : `images/`, `labels/` (train/val), `data.yaml`, `preview/` |
| `runs/detect/<name>/` | courbes, matrice de confusion, `weights/best.pt` |

Choix :
- **Points → boîtes** : YOLO détecte des boîtes, chaque point labelme devient un carré centré de côté
  `POINT_BOX` (0.04 = 4 % de la largeur de l'image, environ 77 px en top et 51 px en poignet).
- **Split par capture** : `top_X` et `wrist_X` (même instant) restent dans le même split, sinon la
  validation est faussée par des quasi-doublons.
- **Augmentation sobre** ([globe_augment.py](globe_augment.py), réglages `AUGMENT` dans `train.py`) : rotation
  ±5° recadrée (jamais de bord gris), zoom avant jusqu'à +30 % sans recul, luminosité ±15 %, saturation ±20 %.
  Ni mosaïque, ni flip (une carte en miroir n'est plus la même carte), ni mixup.

Pistes si la précision est insuffisante : `MODEL=yolov8s.pt`, `IMGSZ=960` (les repères sont petits dans
l'image top en 1920 px), plus d'images pour les classes rares (`47°10'N`, `59°51'N`, `48°28'N`), `POINT_BOX=0.03`.
