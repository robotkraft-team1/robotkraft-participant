#!/bin/bash
# Enregistre un dataset VLA par teleoperation avec 2 cameras, nommées pour
# correspondre EXACTEMENT au checkpoint pre-entraine lerobot/smolvla_base :
#   - camera1 : OAK-D Lite (scene)   -> bridge ZMQ existant (oak_zmq_server.py)
#   - camera2 : webcam USB UVC (poignet) -> backend OpenCV natif de lerobot
#
# Pourquoi camera1/camera2 et pas top/wrist ?
#   Le checkpoint smolvla_base a des input_features "observation.images.camera1/2/3".
#   Le fine-tuning valide que les caméras du dataset sont un sous-ensemble de celles
#   du checkpoint, et le chemin d'inference asynchrone (policy_server, utilise par
#   eval-vla / eval-remote) ne transmet AUCUN rename_map. Si on nommait les caméras
#   top/wrist, le modele ne les reconnaîtrait pas ("All image features are missing").
#   En nommant camera1/camera2, train et inference sont coherents sans rename_map.
#   La 3e vue (camera3) reste absente : avec empty_cameras=0, le modele recoit juste
#   les 2 images presentes, de facon deterministe.
#
# La difference avec record_with_oak.sh :
#   - 2 cameras au lieu d'une (SmolVLA / pi0 generalisent mieux avec 2 vues)
#   - l'instruction textuelle est un param dedie LANG_TASK (anglais), stockee
#     dans le dataset via --dataset.single_task. C'est cette feature "task" que
#     la VLA lit comme entree langage au fine-tuning et a l'inference.
#
# Env vars requises : HF_USER, TASK, LANG_TASK
# Env vars optionnelles : NUM_EPISODES, EPISODE_TIME, RESET_TIME, RESUME,
#                         PUSH_TO_HUB, WRIST_DEV (index ou /dev/videoX de la UVC poignet),
#                         SCENE_WIDTH/SCENE_HEIGHT (resolution OAK, defaut 1280x720),
#                         VCODEC (codec d'encodage, defaut h264)

set -e

# Instruction textuelle en anglais (entree langage de la VLA)
: "${LANG_TASK:?LANG_TASK requis - ex: LANG_TASK=\"pick the red cube and place it in the rack\"}"

# Index ou chemin de la webcam poignet (UVC). Par defaut 0 ; regler apres
# `make detect-cameras` si la UVC n'est pas sur /dev/video0.
WRIST_DEV="${WRIST_DEV:-0}"

# Resolution scene. 1280x720 par defaut : l'encodage AV1 (libsvtav1) en 1920x1080
# est trop gourmand CPU, le record loop stalle (vu a 0.8 Hz) et la camera poignet
# depasse le seuil de fraicheur de 500 ms (read_latest hardcode) -> crash.
# 720p suffit : la VLA redimensionne ses images (224/512) de toute facon.
SCENE_WIDTH="${SCENE_WIDTH:-1280}"
SCENE_HEIGHT="${SCENE_HEIGHT:-720}"

# Codec d'encodage. h264 (libx264) par defaut : bien plus leger que libsvtav1.
# Le service `lerobot` ne monte pas le GPU, h264_nvenc n'est donc pas utilisable
# ici (dispo seulement dans lerobot-gpu).
VCODEC="${VCODEC:-h264}"

# Si WRIST_DEV est un chemin (/dev/videoX), le quoter ; sinon c'est un index int.
case "${WRIST_DEV}" in
  /*) WRIST_IDX="\"${WRIST_DEV}\"" ;;
  *)  WRIST_IDX="${WRIST_DEV}" ;;
esac

# Le bridge ZMQ OAK lit WIDTH/HEIGHT pour sa resolution.
WIDTH="${SCENE_WIDTH}" HEIGHT="${SCENE_HEIGHT}" python3 scripts/camera/oak_zmq_server.py &
ZMQ_PID=$!
trap "kill $ZMQ_PID 2>/dev/null || true" EXIT

echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
sleep 3

# 2 cameras nommees camera1 (scene) + camera2 (poignet) pour matcher le checkpoint.
# camera_name:top reste la clee du message ZMQ publie par oak_zmq_server.py.
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    '--robot.cameras={ camera1: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: '"${SCENE_WIDTH}"', height: '"${SCENE_HEIGHT}"'}, camera2: {type: opencv, index_or_path: '"${WRIST_IDX}"', width: 640, height: 480, fps: 30, color_mode: RGB} }' \
    --dataset.repo_id="${HF_USER}/${TASK}" \
    --dataset.single_task="${LANG_TASK}" \
    --dataset.vcodec="${VCODEC}" \
    --dataset.num_episodes="${NUM_EPISODES:-10}" \
    --dataset.episode_time_s="${EPISODE_TIME:-10}" \
    --dataset.reset_time_s="${RESET_TIME:-10}" \
    --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
    --dataset.streaming_encoding=true \
    --display_data=false \
    --play_sounds=false \
    ${RESUME:+--resume=$RESUME --dataset.root=/workspace/.cache/huggingface/lerobot/$HF_USER/$TASK}
