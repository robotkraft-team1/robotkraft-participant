setup-host:
	@[ "$$(uname)" = "Linux" ] || (echo "setup-host : sans objet sur macOS/Windows (groupe dialout, spécifique Linux). Rien à faire ici." && exit 1)
	@# Debian/Ubuntu : dialout. Arch/Manjaro : uucp (dialout n'existe pas).
	@G=$$(getent group dialout >/dev/null && echo dialout || echo uucp); \
		echo "sudo usermod -aG $$G $(USER)"; sudo usermod -aG $$G $(USER) && \
		echo "Déconnectez-vous et reconnectez-vous pour appliquer le groupe $$G."

# Crée /dev/lerobot_follower et /dev/lerobot_leader (règles udev stables).
# À lancer une seule fois, avec les deux bras disponibles.
setup-udev:
	@[ "$$(uname)" = "Linux" ] || (echo "setup-udev : sans objet sur macOS/Windows (udev est spécifique Linux). En mode natif macOS, passer le port série directement (/dev/cu.usbmodem*), voir README." && exit 1)
	bash scripts/shell/setup_udev.sh

# Crée règle udev pour OAK-D Lite (Luxonis/Movidius 03e7) + symlink /dev/oak. À lancer une seule fois.
setup-oak:
	@[ "$$(uname)" = "Linux" ] || (echo "setup-oak : sans objet sur macOS/Windows (udev est spécifique Linux). Aucune règle à poser, l'OAK-D Lite est détectée directement par depthai." && exit 1)
	echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666", SYMLINK+="oak"' \
		| sudo tee /etc/udev/rules.d/80-movidius.rules
	sudo udevadm control --reload-rules && sudo udevadm trigger
	@echo "Débrancher/rebrancher l'OAK-D Lite puis vérifier : ls -la /dev/oak"

check-oak:
	@test -e /dev/oak || (echo "Erreur: OAK-D Lite non détecté (/dev/oak absent). Brancher et relancer (make setup-oak si première fois)." && exit 1)
	@echo "OAK-D Lite détecté OK"

build:
	docker compose build lerobot

lock:
	docker run --rm -v "$$(pwd):/workspace/hostrepo" -w /workspace/hostrepo robotkraft:latest uv lock

up:
	docker compose up -d lerobot

down:
	docker compose down

shell:
	docker compose run --rm lerobot-base

# Shell avec accès aux deux bras (/dev/lerobot_follower, /dev/lerobot_leader via symlinks udev)
shell-robot:
	docker compose run --rm lerobot bash

# Bascule la variante d'image utilisee par toutes les commandes, en ecrivant
# dans .env. Sans ca, une valeur passee en ligne de commande ne vaut que pour
# la commande en cours.
use-cuda:
	@touch .env && sed -i.bak '/^ROBOTKRAFT_IMAGE=/d' .env && rm -f .env.bak
	@echo 'ROBOTKRAFT_IMAGE=ghcr.io/alsacedigitale/robotkraft:cuda' >> .env
	@echo "Variante CUDA selectionnee. Lance : docker compose pull"

use-cpu:
	@touch .env && sed -i.bak '/^ROBOTKRAFT_IMAGE=/d' .env && rm -f .env.bak
	@echo "Variante CPU selectionnee (defaut). Lance : docker compose pull"

# Dit quelle variante est reellement active et si torch voit un GPU
which-image:
	@echo "image : $${ROBOTKRAFT_IMAGE:-$$(grep -E '^ROBOTKRAFT_IMAGE=' .env 2>/dev/null | cut -d= -f2- || echo 'ghcr.io/alsacedigitale/robotkraft:latest (defaut)')}"
	@docker compose run --rm lerobot-gpu python3 -c "import torch; print('torch :', torch.__version__); print('GPU vu par torch :', torch.cuda.is_available())" 2>/dev/null \
		|| docker compose run --rm lerobot-base python3 -c "import torch; print('torch :', torch.__version__); print('GPU vu par torch :', torch.cuda.is_available())" 2>/dev/null \
		|| echo "(image non tiree)"
