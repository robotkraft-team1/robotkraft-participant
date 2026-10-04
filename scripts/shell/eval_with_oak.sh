#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis lerobot-record piloté par une policy (inférence réelle, pas de teleop)
set -e

source scripts/shell/cameras.sh  # CAMERAS=oak|wrist|both, définit CAMERAS_ARG

# Pendant le reset (ni policy ni teleop), lerobot 0.5.1 tourne sans limite de fps et répète ce warning en boucle
exec 2> >(grep -v --line-buffered "No policy or teleoperator provided" >&2)

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=follower_arm \
    "$CAMERAS_ARG" \
    --policy.path="${POLICY_PATH}" \
    --policy.device="${POLICY_DEVICE:-cpu}" \
    --dataset.repo_id="local/eval_${TASK}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes="${NUM_EPISODES:-5}" \
    --dataset.episode_time_s="${EPISODE_TIME:-30}" \
    --dataset.reset_time_s="${RESET_TIME:-10}" \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.vcodec="${VCODEC:-h264}" \
    ${ENCODER_THREADS:+--dataset.encoder_threads=$ENCODER_THREADS} \
    --display_data=false \
    --play_sounds=false
