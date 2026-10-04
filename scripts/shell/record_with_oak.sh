#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis lerobot-record
# Optionnel : WRIST_DEVICE pour ajouter la caméra poignet (ex. WRIST_DEVICE=0)
set -e

WIDTH="${WIDTH:-640}" HEIGHT="${HEIGHT:-360}" python3 scripts/camera/oak_zmq_server.py &
ZMQ_PID=$!
trap "kill $ZMQ_PID 2>/dev/null || true" EXIT

echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
sleep 3

# Construit le dict de caméras : top (OAK) + wrist (optionnelle)
CAMERAS='top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: 640, height: 360}'
if [ -n "${WRIST_DEVICE}" ]; then
    WRIST_W="${WRIST_WIDTH:-640}"
    WRIST_H="${WRIST_HEIGHT:-480}"
    CAMERAS="${CAMERAS}, wrist: {type: opencv, index_or_path: ${WRIST_DEVICE}, fps: 30, width: ${WRIST_W}, height: ${WRIST_H}}"
    echo "Caméra poignet: /dev/video${WRIST_DEVICE} (${WRIST_W}x${WRIST_H})"
fi

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    "--robot.cameras={ ${CAMERAS} }" \
    --dataset.repo_id="${HF_USER}/${TASK}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes="${NUM_EPISODES:-10}" \
    --dataset.episode_time_s="${EPISODE_TIME:-10}" \
    --dataset.reset_time_s="${RESET_TIME:-10}" \
    --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
    --dataset.streaming_encoding=true \
    --display_data=false \
    --play_sounds=false \
    ${RESUME:+--resume=$RESUME --dataset.root=/workspace/.cache/huggingface/lerobot/$HF_USER/$TASK}
