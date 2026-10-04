#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis lerobot-record piloté par une policy (inférence réelle, pas de teleop)
set -e

TOP_WIDTH="${WIDTH:-640}"
TOP_HEIGHT="${HEIGHT:-480}"
TOP_FPS="${FPS:-30}"
export WIDTH="$TOP_WIDTH"
export HEIGHT="$TOP_HEIGHT"
export FPS="$TOP_FPS"

python3 scripts/camera/oak_zmq_server.py &
ZMQ_PID=$!
trap "kill $ZMQ_PID 2>/dev/null || true" EXIT

echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
sleep 3

if [ "${WITH_WRIST:-false}" = "true" ] || [ -n "${WRIST_DEVICE}" ]; then
    WRIST_DEV="${WRIST_DEVICE:-/dev/video2}"
    CAMERAS="{ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: ${TOP_FPS}, width: ${TOP_WIDTH}, height: ${TOP_HEIGHT}}, wrist: {type: opencv, index_or_path: ${WRIST_DEV}, backend: 200, fps: 30, width: 640, height: 480} }"
    echo "Évaluation bi-caméra activée : top (OAK-D ${TOP_WIDTH}x${TOP_HEIGHT}) + wrist (${WRIST_DEV})"
else
    CAMERAS="{ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: ${TOP_FPS}, width: ${TOP_WIDTH}, height: ${TOP_HEIGHT}} }"
    echo "Évaluation mono-caméra : top (OAK-D ${TOP_WIDTH}x${TOP_HEIGHT})"
fi

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="${CAMERAS}" \
    --policy.path="${POLICY_PATH}" \
    --dataset.repo_id="local/eval_${TASK}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes="${NUM_EPISODES:-5}" \
    --dataset.episode_time_s="${EPISODE_TIME:-30}" \
    --dataset.reset_time_s="${RESET_TIME:-10}" \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --display_data=false \
    --play_sounds=false
