#!/bin/bash
# Téléopération leader -> follower (conteneur lerobot, en arrière-plan) + vidéo OAK-D / poignet avec le modèle
# YOLO en direct (yolo/live.py sur l'hôte). Quitter la fenêtre (Q) ou Ctrl+C arrête aussi la téléopération.
# Lancé par make yolo-teleop, qui fournit LEADER_PORT, FOLLOWER_PORT et les arguments de live.py.
set -u

CONTAINER=robotkraft-yolo-teleop
LOG=yolo/runs/teleop.log
mkdir -p yolo/runs

cleanup() {
    echo "Arrêt de la téléopération..."
    docker stop -t 3 "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# -T : pas de TTY (sinon le processus en arrière-plan est suspendu en attendant le terminal)
docker compose run --rm -T --name "$CONTAINER" lerobot \
    lerobot-teleoperate \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=leader_arm \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=follower_arm \
    >"$LOG" 2>&1 &
TELEOP_PID=$!
echo "Téléopération lancée (logs : $LOG)"

sleep 5
if ! kill -0 "$TELEOP_PID" 2>/dev/null; then
    echo "La téléopération s'est arrêtée au démarrage :"
    tail -20 "$LOG"
    exit 1
fi

yolo/.venv/bin/python yolo/live.py "$@"
