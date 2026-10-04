# OAK-D Perception & Vision Stack (RobotKraft)

Boîte à outils de vision par ordinateur, calibration et détection 3D développée pour le challenge RobotKraft avec la caméra de scène OAK-D Lite et la caméra poignet (InnoMaker U20CAM).

---

## 📁 Architecture des fichiers

```
oak_tools/
├── table_homography.py             # Module central d'homographie (BEV, conversion px <-> mm)
├── calibrate_table.py              # Calibration interactive 4 points de la table
├── demo_homography.py              # Visualisation & validation interactive de l'homographie
├── associate_corners.py            # Association 3D palpeur robot (frames.json) <-> coins 2D (matrice.jpg)
│
├── detect_tubes_3d.py              # Détecteur 3D temps réel des éprouvettes & rack + couleur du liquide
├── yolo_world_3d.py                # Détecteur 3D zero-shot (vocabulaire ouvert) avec OAK-D
├── pointcloud.py                   # Nuage de points 3D haute précision avec Open3D
├── distance.py                     # Mesureur de distance interactif avec filtrage de profondeur & ROI
├── health.py                       # Diagnostic matériel de l'OAK-D (alimentation, température)
│
├── collect_dataset.py              # Capture synchronisée OAK-D (1080p) + Caméra Poignet (1080p)
├── pre_annotate.py                 # Pré-annotation automatique YOLOv8 + galerie HTML d'inspection
├── train_yolo.py                   # Entraînement / fine-tuning YOLOv8 sur éprouvettes et rack
├── benchmark.py                    # Benchmark de performances
│
├── table_homography.json           # Matrice d'homographie calibrée pour la table
├── frames.json                     # Coordonnées 3D enregistrées avec le bras robot (palpeur)
├── matrice.jpg                     # Photo de référence de la table de travail
├── baseline_eval.json              # Métriques d'évaluation de référence
├── evaluation_report.json          # Rapport comparatif des modèles entraînés
├── backup_factory_calibration.json # Sauvegarde de la calibration usine OAK-D
│
└── weights/
    ├── best.pt                     # Meilleur modèle YOLOv8 fine-tuné (tubes & rack)
    ├── best_ft_combined.pt         # Modèle fine-tuné sur dataset combiné
    ├── best_ft_new.pt              # Modèle fine-tuné sur nouvelles captures
    └── best_v1_original.pt         # Poids de référence initiaux
```

---

## 🚀 Utilisation rapide

### 1. Calibration de la table (Homographie)
Pour calibrer la table de travail et convertir les pixels caméra en coordonnées millimétriques réelles :
```bash
python oak_tools/calibrate_table.py --width-mm 800 --height-mm 600
```
Pour tester la calibration existante :
```bash
python oak_tools/demo_homography.py
```

### 2. Détection 3D des éprouvettes et racks
Lance la détection YOLOv8 fine-tunée avec estimation 3D (X, Y, Z en mm) et classification des couleurs (Jaune, Cyan, Magenta, Transparent) :
```bash
python oak_tools/detect_tubes_3d.py
```

### 3. Mesure de distance & profondeur interactive
Lance le visualiseur de profondeur haute précision avec sélection ROI et statistiques :
```bash
python oak_tools/distance.py
```

### 4. Collecte de données multi-caméras
Capture des photos synchronisées scène (OAK-D) et poignet (InnoMaker) :
```bash
python oak_tools/collect_dataset.py
```
- `[ESPACE]` : Photo simultanée sur les 2 caméras
- `[o]` : Photo OAK-D seule
- `[w]` : Photo poignet seule
