#!/bin/bash
# Evaluation d'une policy VLA (SmolVLA / pi0) sur le robot reel, avec la policy
# chargee sur un serveur d'inference distant (make policy-server sur le GPU).
# Robot + cameras restent en local ; seule l'inference est distante.
#
# Les caméras sont nommees camera1 (scene OAK-D) et camera2 (poignet UVC) pour
# correspondre aux input_features du checkpoint lerobot/smolvla_base. Le chemin
# d'inference asynchrone (robot_client -> policy_server) ne transmet AUCUN
# rename_map : nommer les caméras comme le checkpoint est donc obligatoire pour
# que le modele les reconnaisse.
#
# Convention (identique a record-vla) :
#   TASK      = nom du dataset (sert a trouver le checkpoint : HF_USER/TASK)
#   LANG_TASK = l'ordre textuel en anglais envoye a la VLA (--task=)
# Si LANG_TASK n'est pas fourni, il vaut par defaut TASK.
#
# Env vars requises : TASK (nom dataset), SERVER (ip:port du policy_server)
# Env vars optionnelles :
#   LANG_TASK          : instruction anglaise (defaut: TASK)
#   POLICY_TYPE        : smolvla (defaut) | pi0 | pi05
#   PRETRAINED_PATH    : repo HF (HF_USER/TASK) ou chemin local du checkpoint
#   HF_USER            : pour deriv PRETRAINED_PATH par defaut
#   CHECKPOINT         : nom du checkpoint local (defaut last)
#   ACTIONS_PER_CHUNK  : nombre d'actions consommees par chunk (defaut 100)
#   CHUNK_SIZE_THRESHOLD: seuil de re-inference (defaut 0)
#   DEBUG_QUEUE        : true -> visualise la taille de la file d'actions
#   WRIST_DEV          : index ou /dev/videoX de la UVC poignet (defaut 0)
#   SCENE_WIDTH/SCENE_HEIGHT : resolution OAK (defaut 1280x720, comme record-vla :
#                            1920x1080 est trop gourmand CPU pendant l'inference)

set -e

: "${TASK:?TASK requis (nom du dataset) - ex: TASK=tri_cubes}"
: "${SERVER:?SERVER requis - ex: SERVER=10.0.0.5:8080 (ip:port du policy_server)}"

POLICY_TYPE="${POLICY_TYPE:-smolvla}"
# L'ordre textuel : par defaut on reprend le nom du dataset, sinon LANG_TASK.
LANG_TASK="${LANG_TASK:-$TASK}"
# Checkpoint : celui passe en arg, sinon HF_USER/TASK si HF_USER defini,
# sinon le checkpoint local outputs/smolvla_TASK.
CHECKPOINT="${CHECKPOINT:-last}"
if [ -z "${PRETRAINED_PATH:-}" ]; then
    if [ -n "${HF_USER:-}" ]; then
        PRETRAINED_PATH="${HF_USER}/${TASK}"
    else
        PRETRAINED_PATH="outputs/smolvla_${TASK}/checkpoints/${CHECKPOINT}/pretrained_model"
    fi
fi
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-100}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0}"
WRIST_DEV="${WRIST_DEV:-0}"

# Resolution scene : 1280x720 par defaut, coherent avec record-vla. Le 1920x1080
# est inutile (le client resize les images a la resolution du checkpoint) et
# charge le CPU inutilement pendant l'inference.
SCENE_WIDTH="${SCENE_WIDTH:-1280}"
SCENE_HEIGHT="${SCENE_HEIGHT:-720}"

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

RUN_CMD=(python scripts/robot/robot_client_zmq.py
    --robot.type=so101_follower
    --robot.port=/dev/ttyACM0
    --robot.id=follower_arm
    '--robot.cameras={ camera1: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: '"${SCENE_WIDTH}"', height: '"${SCENE_HEIGHT}"'}, camera2: {type: opencv, index_or_path: '"${WRIST_IDX}"', width: 640, height: 480, fps: 30, color_mode: RGB} }'
    --task="${LANG_TASK}"
    --server_address="${SERVER}"
    --policy_type="${POLICY_TYPE}"
    --pretrained_name_or_path="${PRETRAINED_PATH}"
    --policy_device=cuda
    --client_device=cpu
    --actions_per_chunk="${ACTIONS_PER_CHUNK}"
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}")

# DEBUG_QUEUE=true -> affiche la taille de file d'actions en direct, pour repérer
# si les ratés coïncident avec une frontiere de chunk (file vide -> re-inference)
if [ "${DEBUG_QUEUE:-false}" = "true" ]; then
    RUN_CMD+=(--debug_visualize_queue_size=true)
fi

"${RUN_CMD[@]}"
