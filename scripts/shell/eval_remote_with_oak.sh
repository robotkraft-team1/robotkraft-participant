#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis RobotClient piloté par un PolicyServer distant
set -e

source scripts/shell/cameras.sh  # CAMERAS=oak|wrist|both, définit CAMERAS_ARG

RUN_CMD=(python scripts/robot/robot_client_zmq.py
    --robot.type=so101_follower
    --robot.port="$FOLLOWER_PORT"
    --robot.id=follower_arm
    "$CAMERAS_ARG"
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
