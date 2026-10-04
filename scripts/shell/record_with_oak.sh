#!/bin/bash
# Lance lerobot-record (teleop) avec les caméras choisies par CAMERAS (voir cameras.sh)
set -e

source scripts/shell/cameras.sh  # CAMERAS=oak|wrist|both, définit CAMERAS_ARG

lerobot-record \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=leader_arm \
    "$CAMERAS_ARG" \
    --dataset.repo_id="${HF_USER}/${TASK}" \
    --dataset.single_task="${TASK}" \
    --dataset.num_episodes="${NUM_EPISODES:-10}" \
    --dataset.episode_time_s="${EPISODE_TIME:-10}" \
    --dataset.reset_time_s="${RESET_TIME:-10}" \
    --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
    --dataset.streaming_encoding=true \
    --dataset.vcodec="${VCODEC:-h264}" \
    ${ENCODER_THREADS:+--dataset.encoder_threads=$ENCODER_THREADS} \
    --display_data=false \
    --play_sounds=false \
    ${RESUME:+--resume=$RESUME --dataset.root=/workspace/.cache/huggingface/lerobot/$HF_USER/$TASK}
