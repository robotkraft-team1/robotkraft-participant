#!/bin/bash
# À sourcer : choisit les caméras pour lerobot-record / eval selon CAMERAS, lance le serveur ZMQ OAK
# si nécessaire, et définit CAMERAS_ARG (argument --robot.cameras=...).
#   CAMERAS=oak    (défaut) OAK-D Lite seule, nommée "top"
#   CAMERAS=wrist  caméra poignet USB (U20CAM, ET-S231) seule, nommée "wrist"
#   CAMERAS=both   les deux
# Caméra poignet : WRIST_DEVICE (défaut /dev/video2), WRIST_WIDTH x WRIST_HEIGHT (défaut 640x480), 30 fps MJPG.
# OAK-D Lite : OAK_WIDTH x OAK_HEIGHT (défaut 640x360). Le 1920x1080 sature le conteneur et fait expirer la lecture des servos.
# ⚠️ Une policy doit être évaluée avec exactement les mêmes CAMERAS que celles de son dataset.

CAMERAS="${CAMERAS:-oak}"
WRIST_DEVICE="${WRIST_DEVICE:-/dev/video2}"
export OAK_WIDTH="${OAK_WIDTH:-640}" OAK_HEIGHT="${OAK_HEIGHT:-360}"  # lus aussi par oak_zmq_server.py
OAK_CAM="top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: ${OAK_WIDTH}, height: ${OAK_HEIGHT}}"
WRIST_CAM="wrist: {type: opencv, index_or_path: ${WRIST_DEVICE}, fps: 30, width: ${WRIST_WIDTH:-640}, height: ${WRIST_HEIGHT:-480}, fourcc: MJPG, backend: 200}"  # 200 = V4L2 : avec ANY, OpenCV ignore MJPG et la résolution

case "$CAMERAS" in
    oak)   CAMS="$OAK_CAM" ;;
    wrist) CAMS="$WRIST_CAM" ;;
    both)  CAMS="$OAK_CAM, $WRIST_CAM" ;;
    *) echo "CAMERAS=$CAMERAS invalide (oak, wrist ou both)"; exit 1 ;;
esac
CAMERAS_ARG="--robot.cameras={ $CAMS }"

if [ "$CAMERAS" != "oak" ] && [ ! -e "$WRIST_DEVICE" ]; then
    echo "Caméra poignet introuvable : $WRIST_DEVICE (voir v4l2-ctl --list-devices sur l'hôte, puis WRIST_DEVICE=/dev/videoX)"
    exit 1
fi

if [ "$CAMERAS" != "wrist" ]; then
    python3 scripts/camera/oak_zmq_server.py &
    ZMQ_PID=$!
    trap "kill $ZMQ_PID 2>/dev/null || true" EXIT
    echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
    sleep 3
fi
echo "Caméras : $CAMERAS"
