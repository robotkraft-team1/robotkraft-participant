# Détecte l'OAK-D Lite et teste un frame RGB dans le container
detect-oak: check-oak
	docker compose run --rm lerobot-camera python3 scripts/camera/oak_detect.py

# Détecte les caméras disponibles dans le container
detect-cameras:
	docker compose run --rm lerobot python3 scripts/camera/camera_detect.py

# Preview live webcam USB générique (ET-S231 et similaires) - exécuté en direct sur l'hôte, pas de docker (évite de tirer toute la stack ML pour une simple webcam)
# Usage : make view-camera [DEVICE=/dev/video0] [WIDTH=1920 HEIGHT=1080 FPS=30]
view-camera:
	DEVICE=$(or $(DEVICE),/dev/video0) \
	WIDTH=$(or $(WIDTH),1920) \
	HEIGHT=$(or $(HEIGHT),1080) \
	FPS=$(or $(FPS),30) \
	python3 scripts/camera/camera_viewer.py

# Prend une photo et la sauvegarde dans datasets/
# Usage : make photo [FILE=datasets/test/frame.jpg] [CROP_X=0 CROP_Y=160 CROP_W=600 CROP_H=320]
photo:
	docker compose run --rm \
		-e OUTPUT_FILE=/workspace/$(or $(FILE),datasets/photo_$(shell date +%Y%m%d_%H%M%S).jpg) \
		-e CROP_X=$(or $(CROP_X),0) \
		-e CROP_Y=$(or $(CROP_Y),0) \
		-e CROP_W=$(or $(CROP_W),0) \
		-e CROP_H=$(or $(CROP_H),0) \
		lerobot-camera python3 scripts/camera/capture_photo.py

# Calibration balance des blancs (offsets utilisés par make photo, PAS par lerobot-record)
# Usage : make calibrate-wb (OAK-D Lite)  |  make calibrate-wb DEVICE=/dev/video2 (caméra poignet USB)
calibrate-wb:
	xhost +local:docker
	docker compose run --rm -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix \
		$(if $(DEVICE),-e WB_DEVICE=$(DEVICE) lerobot-camera python3 scripts/camera/calibrate_wb.py,lerobot-camera bash scripts/shell/calibrate_with_oak.sh calibrate_wb.py)

# Dataset d'images du globe (annotation puis YOLO) : une image OAK-D + une image poignet par capture, + articulations du follower
# Usage : make capture-globe [NAME=globe] [CAMERAS=both|oak|wrist] [ARM=teleop|read|none] [AUTO_INTERVAL=1.0]
#   ARM=teleop : on vise avec le leader (défaut) ; ARM=read : follower sans couple, bougé à la main ; ARM=none : pas de bras
#   Fenêtre : ESPACE = capture, A = capture auto on/off, Q = quitter. Sortie : datasets/$(NAME)/images + captures.csv
capture-globe:
	@xhost +local:docker >/dev/null 2>&1 || true
	docker compose run --rm \
		-e OUT_DIR=datasets/$(or $(NAME),globe) \
		-e CAMERAS=$(or $(CAMERAS),both) \
		-e ARM=$(or $(ARM),teleop) \
		-e WRIST_DEVICE=$(WRIST_DEVICE) \
		-e AUTO_INTERVAL=$(or $(AUTO_INTERVAL),1.0) \
		$(if $(N_SHOTS),-e N_SHOTS=$(N_SHOTS)) \
		-e HOST_UID=$(shell id -u) -e HOST_GID=$(shell id -g) -v /etc/localtime:/etc/localtime:ro \
		-w /workspace/scripts/camera -e PYTHONPATH=/workspace/scripts/camera \
		lerobot sh -c 'cd /workspace && python3 scripts/camera/capture_globe.py'

# Label Studio (annotation des images pour YOLO) sous Docker : http://localhost:8080
# Les images de datasets/ sont accessibles via Settings > Cloud Storage > Add Source Storage > Local files,
# chemin absolu /label-studio/files/<dossier>/images (ex. /label-studio/files/globe/images).
# Projets et annotations conservés dans .label-studio/ (à côté du repo). Export : format "YOLO".
label-studio:
	@mkdir -p .label-studio
	docker run --rm -it -p 8080:8080 \
		--user $(shell id -u):$(shell id -g) \
		-v $(CURDIR)/.label-studio:/label-studio/data \
		-v $(CURDIR)/datasets:/label-studio/files:ro \
		-e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
		-e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files \
		heartexlabs/label-studio:latest
