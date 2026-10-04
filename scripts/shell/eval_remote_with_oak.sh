#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis RobotClient piloté par un PolicyServer distant
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

RUN_CMD=(python scripts/robot/robot_client_zmq.py
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM0
    --robot.id=follower_arm
    "--robot.cameras={ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: ${TOP_FPS}, width: ${TOP_WIDTH}, height: ${TOP_HEIGHT}} }"
    --task="${TASK}"
    --server_address="${SERVER_ADDRESS}"
    --policy_type=act
    --pretrained_name_or_path="${PRETRAINED_PATH}"
    --policy_device=cuda
    --client_device=cpu
    --actions_per_chunk="${ACTIONS_PER_CHUNK:-100}"
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD:-0}")

# DEBUG_QUEUE=true -> affiche taille de queue d'actions en direct, pour repérer si les ratés coïncident avec une frontière de chunk (queue vide -> re-inférence)
if [ "${DEBUG_QUEUE:-false}" = "true" ]; then
    RUN_CMD+=(--debug_visualize_queue_size=true)
fi

"${RUN_CMD[@]}"
