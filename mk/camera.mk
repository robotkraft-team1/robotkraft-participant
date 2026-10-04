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

# Calibration balance des blancs
calibrate-wb:
	xhost +local:docker
	docker compose run --rm -e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix lerobot-camera bash scripts/shell/calibrate_with_oak.sh calibrate_wb.py

# Preview live OAK-D Lite (dans le container, via X11 ; lance oak_zmq_server.py puis le viewer)
# Usage : make view-oak [WIDTH=640 HEIGHT=480 FPS=30]
view-oak:
	xhost +local:docker
	docker compose run --rm \
		-e DISPLAY=$(DISPLAY) -v /tmp/.X11-unix:/tmp/.X11-unix \
		-e WIDTH=$(or $(WIDTH),640) -e HEIGHT=$(or $(HEIGHT),480) -e FPS=$(or $(FPS),30) \
		lerobot-camera bash scripts/shell/calibrate_with_oak.sh oak_viewer.py
