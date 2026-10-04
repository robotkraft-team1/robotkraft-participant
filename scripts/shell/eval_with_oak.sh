#!/bin/bash
# Lance le serveur ZMQ OAK en arrière-plan puis lerobot-record piloté par une policy (inférence réelle, pas de teleop)
set -e

WIDTH="${WIDTH:-640}" HEIGHT="${HEIGHT:-360}" python3 scripts/camera/oak_zmq_server.py &
ZMQ_PID=$!
trap "kill $ZMQ_PID 2>/dev/null || true" EXIT

echo "ZMQ server PID=$ZMQ_PID, attente 3s..."
sleep 3

# Compatibilité version : certains checkpoints sont sauvegardés par une version de lerobot
# plus récente que le conteneur (0.5.1) qui écrit un champ 'pretrained_revision' absent de
# PreTrainedConfig ici. draccus refuse alors le champ et l'éval échoue au chargement du
# modèle. On le retire du config.json local s'il est présent (sans effet : il vaut null).
if [ -f "${POLICY_PATH}/config.json" ]; then
    python3 -c "
import json, sys
p = '${POLICY_PATH}/config.json'
d = json.load(open(p))
if 'pretrained_revision' in d:
    del d['pretrained_revision']
    json.dump(d, open(p, 'w'), indent=4)
    print('Nettoyage: champ pretrained_revision retiré de', p)
"
fi

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    '--robot.cameras={ top: {type: zmq, server_address: localhost, port: 5555, camera_name: top, fps: 30, width: 640, height: 360} }' \
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
