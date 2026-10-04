# Ces cibles nécessitent make setup-udev (une fois) pour créer /dev/lerobot_*
# Service lerobot (privileged) : le conteneur voit les ttyACM* de l'hôte sous leur nom, on lui passe donc
# le vrai port de chaque bras, résolu depuis les symlinks udev (ttyACM0 = follower n'est vrai que si le
# follower a été branché en premier). Sans symlinks : ttyACM0/1 comme avant. Surchargeable (FOLLOWER_PORT=...).
export FOLLOWER_PORT ?= $(or $(realpath /dev/lerobot_follower),/dev/ttyACM0)
export LEADER_PORT ?= $(or $(realpath /dev/lerobot_leader),/dev/ttyACM1)
FOLLOWER = $(FOLLOWER_PORT)
LEADER = $(LEADER_PORT)

# Services mono-bras (lerobot-follower/-leader) : périphérique hôte monté en ttyACM0/1. Symlink udev si présent,
# pour ne pas dépendre de l'ordre de branchement ; sinon ttyACM0 (follower) / ttyACM1 (leader) comme avant.
export FOLLOWER_DEV ?= $(if $(wildcard /dev/lerobot_follower),/dev/lerobot_follower,/dev/ttyACM0)
export LEADER_DEV ?= $(if $(wildcard /dev/lerobot_leader),/dev/lerobot_leader,/dev/ttyACM1)

# Les cibles calibrate-follower et calibrate-leader sont à lancer une seule fois pour chaque bras pour lancer la calibration le leader et le follower
calibrate-follower:
	docker compose run --rm lerobot lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER) \
		--robot.id=follower_arm

calibrate-leader:
	docker compose run --rm lerobot lerobot-calibrate \
		--teleop.type=so101_leader \
		--teleop.port=$(LEADER) \
		--teleop.id=leader_arm

teleop:
	docker compose run --rm lerobot \
		lerobot-teleoperate \
		--teleop.type=so101_leader \
		--teleop.port=$(LEADER) \
		--teleop.id=leader_arm \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER) \
		--robot.id=follower_arm

# Vérifie que /dev/lerobot_follower et /dev/lerobot_leader répondent dans le container
check-devices:
	docker compose run --rm lerobot python3 scripts/robot/check_devices.py

# Requiert robots branchés, vérifie les ids des servo-moteurs
scan-motors:
	docker compose run --rm lerobot python3 scripts/robot/scan_motors.py

# Vérifie le voltage de chaque servo (un bras à la fois, sans symlink udev)
check-voltage-follower:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower python3 scripts/robot/diag_voltage.py

check-voltage-leader:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM1 lerobot-leader python3 scripts/robot/diag_voltage.py

check-voltage: check-voltage-follower check-voltage-leader

# Lancer un script avec accès direct au follower (sans symlink udev)
# Usage : make script-follower FILE=scripts/robot/move_central_position.py
script-follower:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower python3 $(FILE)

# Lancer un script avec accès direct au leader (sans symlink udev)
# Usage : make script-leader FILE=scripts/robot/move_central_position.py
script-leader:
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM1 lerobot-leader python3 $(FILE)
