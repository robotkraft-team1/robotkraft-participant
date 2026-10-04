# Détection YOLOv8 sur le globe (annotations labelme -> dataset YOLO -> entraînement), exécuté sur l'hôte
# dans un venv dédié yolo/.venv (indépendant de l'image lerobot ; utilise le GPU NVIDIA s'il y en a un).
YOLO_PY = yolo/.venv/bin/python
, := ,

# Crée le venv et installe ultralytics (une seule fois)
yolo-setup:
	uv venv --allow-existing --python 3.12 yolo/.venv
	uv pip install --python $(YOLO_PY) -r yolo/requirements.txt
	# depthai (OAK-D Lite) pour yolo-live, même version et même index que le conteneur (pyproject.toml)
	uv pip install --python $(YOLO_PY) "depthai==3.7.1" \
		--extra-index-url https://artifacts.luxonis.com/artifactory/luxonis-python-release-local/

# JSON labelme (yolo/datasets/$(NAME)/images) -> yolo/data/$(NAME) (split train/val par capture, data.yaml)
# Usage : make yolo-convert [NAME=globe] [YOLO_TASK=detect|segment] [YOLO_CAMERAS=top,wrist] [VAL=0.2] [POINT_BOX=0.04] [PREVIEW=1]
yolo-convert:
	$(YOLO_PY) yolo/labelme2yolo.py \
		--src $(or $(SRC),yolo/datasets/$(or $(NAME),globe)/images) \
		--out yolo/data/$(or $(NAME),globe) \
		--task $(or $(YOLO_TASK),detect) \
		--cameras $(or $(YOLO_CAMERAS),top$(,)wrist) \
		--val $(or $(VAL),0.2) \
		--point-box $(or $(POINT_BOX),0.04) \
		$(if $(PREVIEW),--preview)

# Usage : make yolo-train [NAME=globe] [MODEL=yolov8s.pt] [EPOCHS=100] [IMGSZ=640] [BATCH=16] [RESUME=1]
yolo-train:
	$(YOLO_PY) yolo/train.py \
		--data yolo/data/$(or $(NAME),globe)/data.yaml \
		--name $(or $(NAME),globe) \
		--epochs $(or $(EPOCHS),100) \
		--imgsz $(or $(IMGSZ),640) \
		--batch $(or $(BATCH),16) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(RESUME),--resume)

# Planches d'images augmentées (mêmes réglages que yolo-train) avec leurs boîtes ; ESPACE = suivant, S = sauver, Q = quitter
# Usage : make yolo-preview-aug [NAME=globe] [SPLIT=train|val] [IMGSZ=640] [SAVE_ONLY=1]
yolo-preview-aug:
	$(YOLO_PY) yolo/preview_augment.py \
		--data yolo/data/$(or $(NAME),globe)/data.yaml \
		--split $(or $(SPLIT),train) \
		--imgsz $(or $(IMGSZ),640) \
		$(if $(SAVE_ONLY),--no-show)

# Inférence pour vérifier à l'œil (résultats dans yolo/runs/predict*)
# Usage : make yolo-predict WEIGHTS=yolo/runs/detect/globe/weights/best.pt [SOURCE=yolo/datasets/globe/images] [CONF=0.25]
yolo-predict:
	cd yolo && .venv/bin/yolo predict \
		model=$(abspath $(or $(WEIGHTS),yolo/runs/detect/$(or $(NAME),globe)/weights/best.pt)) \
		source=$(abspath $(or $(SOURCE),yolo/data/$(or $(NAME),globe)/images/val)) \
		conf=$(or $(CONF),0.25) project=runs name=predict save=True

# Arguments communs de yolo/live.py (yolo-live, yolo-teleop). Caméras : OAK-D Lite (top) + poignet (WRIST_DEVICE, /dev/video2)
YOLO_LIVE_ARGS = \
	--weights $(or $(WEIGHTS),yolo/runs/detect/$(or $(NAME),globe)/weights/best.pt) \
	--cameras $(or $(CAMERAS),both) \
	--wrist-device $(or $(WRIST_DEVICE),/dev/video2) \
	--imgsz $(or $(IMGSZ),640) \
	--conf $(or $(CONF),0.25) \
	$(if $(TARGET),--target "$(TARGET)")

# Vidéo des caméras avec le modèle en direct, sans bras. Fenêtre : Q = quitter, S = snapshot, ESPACE = pause, +/- = confiance
# Usage : make yolo-live [CAMERAS=both|oak|wrist] [WRIST_DEVICE=/dev/video2] [WEIGHTS=...best.pt] [CONF=0.25] [TARGET="48°28"]
yolo-live: check-oak
	$(YOLO_PY) yolo/live.py $(YOLO_LIVE_ARGS)

# Téléopération leader -> follower (Docker) + vidéo des caméras avec le modèle en direct (hôte).
# Quitter la fenêtre (Q) ou Ctrl+C arrête aussi la téléopération. Logs du bras : yolo/runs/teleop.log
# Usage : make yolo-teleop [mêmes options que yolo-live]
yolo-teleop: check-oak
	yolo/teleop_live.sh $(YOLO_LIVE_ARGS)
