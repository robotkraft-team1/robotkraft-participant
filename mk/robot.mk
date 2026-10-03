# Ces cibles nécessitent make setup-udev (une fois) pour créer /dev/lerobot_*

# Les cibles calibrate-follower et calibrate-leader sont à lancer une seule fois pour chaque bras pour lancer la calibration le leader et le follower
calibrate-follower:
	docker compose run --rm lerobot lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=/dev/ttyACM0 \
		--robot.id=follower_arm

calibrate-leader:
	docker compose run --rm lerobot lerobot-calibrate \
		--teleop.type=so101_leader \
		--teleop.port=/dev/ttyACM1 \
		--teleop.id=leader_arm

teleop:
	docker compose run --rm lerobot \
		lerobot-teleoperate \
		--teleop.type=so101_leader \
		--teleop.port=/dev/ttyACM1 \
		--teleop.id=leader_arm \
		--robot.type=so101_follower \
		--robot.port=/dev/ttyACM0 \
		--robot.id=follower_arm

# Vérifie que ttyACM0=follower et ttyACM1=leader dans le container
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

# Cinématique inverse / commande de position du follower (variante kinematics).
# Dry-run par défaut : afficher les ticks sans bouger le bras ne nécessite aucun
# argument. Pour déplacer le bras : ARGS='--execute' (le bras bouge vraiment).
# Usage :
#   make ik ARGS='--point 0.30 -0.10 0.25'
#   make ik ARGS='--point 0.30 -0.10 0.25 --execute'
#   make ik ARGS='--joints shoulder_lift=-20 elbow_flex=30'
ik:
	@[ "$${ROBOTKRAFT_IMAGE}" = "robotkraft:kinematics" ] || grep -q '^ROBOTKRAFT_IMAGE=robotkraft:kinematics' .env 2>/dev/null \
		|| { echo "Variante kinematics requise : make use-kinematics && make build-kinematics"; exit 1; }
	# lerobot-follower (un seul bras, /dev/ttyACM0) : IK ne pilote que le follower,
	# pas le service lerobot qui monte les deux bras (/dev/lerobot_follower + _leader).
	docker compose run --rm -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower \
		python3 scripts/kinematics/ik.py $(ARGS)

# Visualisation 3D du bras (meshcat, navigateur). Le port 7000 est publié, le
# modèle s'ouvre sur http://localhost:7000.
#   make viz ARGS='--sequence scripts/kinematics/sequence_demo.json'   (cibles, dry-run)
#   make viz ARGS='--point 0.30 -0.10 0.25'
#   make watch-arms   le bras virtuel suit le bras réel (bras branché, lecture seule)
viz:
	@[ "$${ROBOTKRAFT_IMAGE}" = "robotkraft:kinematics" ] || grep -q '^ROBOTKRAFT_IMAGE=robotkraft:kinematics' .env 2>/dev/null \
		|| { echo "Variante kinematics requise : make use-kinematics && make build-kinematics"; exit 1; }
	docker compose run --rm -p 7000:7000 lerobot-follower \
		python3 scripts/kinematics/visualize.py $(ARGS)

watch-arms:
	@[ "$${ROBOTKRAFT_IMAGE}" = "robotkraft:kinematics" ] || grep -q '^ROBOTKRAFT_IMAGE=robotkraft:kinematics' .env 2>/dev/null \
		|| { echo "Variante kinematics requise : make use-kinematics && make build-kinematics"; exit 1; }
	docker compose run --rm -p 7000:7000 -e ROBOT_PORT=/dev/ttyACM0 lerobot-follower \
		python3 scripts/kinematics/visualize.py --watch
