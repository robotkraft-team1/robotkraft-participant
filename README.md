# RobotKraft, version participants

> Base technique du challenge RobotKraft : environnement Docker prêt à l'emploi, détection des
> bras et des caméras, diagnostic matériel, et raccourcis vers les commandes LeRobot
> (téléopération, enregistrement de datasets, entraînement, évaluation en local ou sur GPU
> distant).
>
> La perception et la stratégie ne sont pas fournies : c'est ce que le challenge évalue, à vous
> de l'écrire.
>
> Les informations propres à chaque édition (lieu, horaires, règlement, épreuves, entraide)
> sont communiquées à part.

Robotique spatiale/chimie pour hackathon IA & Robotique : bras SO-ARM101 (leader/follower), imitation learning avec [LeRobot](https://huggingface.co/docs/lerobot), le tout conteneurisé sous Docker.

---

## Sécurité : ordre de branchement

⚠️ Deux erreurs détruisent du matériel.

**L'alimentation d'abord, l'USB ensuite.** USB branché avant l'alimentation, le port USB alimente la carte et peut griller. Ordre : alimentation, puis USB follower, puis USB leader. Pour débrancher, l'inverse exact : USB leader, USB follower, puis enfin alimentation.

**Chaque bras a sa tension et les connecteurs sont interchangeables.** Follower 12V, leader 5V. Le 12V sur un leader le détruit. Repérage par couleur : vert = 5V = leader, blanc = 12V = follower. Le fil vert sur le fil vert, le fil blanc sur le fil blanc.

- Erreur `input voltage error` sous charge (~5,4V) : alimentation insuffisante.
- Ports `/dev/ttyACM0`/`ACM1` s'attribuent selon l'ordre de branchement (follower en premier = `ACM0`). Vérifier `ls /dev/ttyACM*` avant chaque session.
- Les deux bras ont le même VID:PID USB (puce WCH), donc indistinguables par ça : en cas de doute sur qui est qui, se fier à la tension (follower 12V, leader 5V), pas au numéro de série ni à l'ordre de branchement. `python3 scripts/robot/identify_arms.py` l'automatise (Linux et macOS).

---

## Architecture

| Composant               |
|-------------------------|
| SO-ARM101               |
| LeRobot (HuggingFace)   |
| Caméra poignet : InnoMaker U20CAM ou ET-S231 selon le poste (toutes deux USB UVC) |
| Caméra de scène (optionnelle, 11 disponibles sur place) : OAK-D Lite (depthai) |
| Python 3.12.9 + uv      |
| Docker                  |

---

## Services Docker

| Service |
|---------|
| `lerobot-base` |
| `lerobot` |
| `lerobot-gpu` |
| `lerobot-follower` |
| `lerobot-leader` |
| `lerobot-camera` |

> `lerobot` requiert `/dev/lerobot_follower` **et** `/dev/lerobot_leader`. Un seul bras branché : `lerobot-follower` / `lerobot-leader` (montent `/dev/ttyACM0`/`ACM1`, sans symlink udev).

---

## Prérequis

- [Docker](https://www.docker.com/)
- `HF_TOKEN` dans `.env` (voir `.env.example`) pour push/pull datasets et checkpoints HuggingFace (obtenable via Access Token > Create New Token sur le site de HuggingFace)

---

## Installation

L'image est prête, il n'y a rien à compiler. Tire-la plutôt que de la construire, tu gagnes
une vingtaine de minutes et tu économises la connexion du lieu :

```bash
docker compose pull      # récupère l'image depuis ghcr.io
```

**Fais-le chez toi avant de venir.** L'image pèse environ 2,7 Go, et des dizaines de
téléchargements simultanés sur la connexion du lieu, ce n'est pas une bonne idée.

Si ton portable a un GPU NVIDIA et que tu veux entraîner en local, bascule sur la
variante CUDA (environ 9 Go) :

```bash
make use-cuda            # écrit le choix dans .env, donc il persiste
docker compose pull
make which-image         # vérifie : doit afficher GPU vu par torch : True
```

Sinon, l'entraînement se fait sur les serveurs GPU distants, et la variante par
défaut suffit. `make use-cpu` revient en arrière.

⚠️ Ne passe pas `ROBOTKRAFT_IMAGE=...` directement devant une commande : la valeur
ne vaut que pour cette commande, et `make train` repartirait sur la variante CPU
en entraînant sur processeur sans rien signaler.

### Régler ta machine, obligatoire même si tu tires l'image

Ces trois commandes configurent ton système, pas l'image. Sans elles, les bras et
la caméra ne seront pas visibles depuis les conteneurs.

```bash
make setup-host # Groupe dialout (déconnecte puis reconnecte ta session juste après)
make setup-udev # Règles udev et symlinks /dev/lerobot_follower et _leader
make setup-oak  # Règle udev pour la caméra OAK-D Lite (une seule fois)
```

Copie aussi le fichier de configuration :

```bash
cp .env.example .env    # puis renseigne HF_TOKEN
```

### Reconstruire l'image, seulement si tu modifies le Dockerfile ou les dépendances

```bash
make build
```

### ⛔ Ne reconfigure pas les servomoteurs

Les identifiants des servos sont **déjà définis sur les bras qu'on te prête**, et
toute la chaîne en dépend. Les redéfinir casse le bras pour toi et pour l'équipe
qui l'utilisera après toi, et il faut ensuite le reconfigurer servo par servo.

**Ne lance jamais `lerobot-setup-motors`, ni aucune commande de configuration des
identifiants moteurs.** Si un servo semble muet ou mal numéroté, ce n'est pas à toi
de le réparer : viens voir un coach.

Pour vérifier que les servos répondent, sans rien modifier :

```bash
make scan-motors    # liste les identifiants vus sur le bus
make check-voltage  # tension de chaque servo
```

### Calibration des robots

```bash
make calibrate-follower # Calibration follower (une fois)
make calibrate-leader   # Calibration leader (une fois)
```

> Calibrations stockées dans `.cache/` (volume Docker). Pour recalibrer : supprimer le `.json` correspondant dans `.cache/huggingface/lerobot/calibration/`.

### Vérification après installation

```bash
ls /dev/ttyACM*    # follower/leader détectés
make check-devices # vérifie ACM0=follower, ACM1=leader
make check-voltage # tension de chaque servo (follower + leader)
```

### Caméra poignet : InnoMaker U20CAM ou ET-S231 selon le poste

- Caméra USB UVC standard (U20CAM sur les kits 1-8, ET-S231 intégrée sur les kits 9-11), aucune règle udev à poser
- `make detect-cameras` -> liste les caméras USB disponibles dans le container
- `make view-camera DEVICE=/dev/videoX` -> preview live sur l'hôte, pour trouver le bon device
- ET-S231 (kits 9-11) : mise au point MANUELLE, par la bague de l'objectif. Avant d'enregistrer des démonstrations, vérifiez dans un aperçu caméra (`make view-camera DEVICE=/dev/videoX` sous Linux, ou l'application caméra du système sous macOS/Windows) que l'image est nette à la distance où le bras saisit les objets, puis ne touchez plus à la bague : un réglage qui bouge en cours de route rend vos données d'entraînement incohérentes.

### Caméra de scène (optionnelle, 11 disponibles sur place) : OAK-D Lite

- **USB3 obligatoire**
- `make setup-oak` une seule fois (règle udev + symlink `/dev/oak`)
- `make detect-oak` -> doit capturer un frame RGB pour valider la détection de la caméra
- `make view-oak [WIDTH=640 HEIGHT=480 FPS=30]` -> preview live (fenêtre X11), `Q`/Échap pour quitter
- **Ne jamais** `import depthai` depuis l'hôte, **ni** `docker run` (toujours `docker compose run lerobot-camera ...`) : un container éphémère peut garder le device verrouillé
- En mode natif (sans Docker, cf. section dédiée) : deux terminaux, `WIDTH=640 HEIGHT=480 python3 scripts/camera/oak_zmq_server.py` puis `python3 scripts/camera/oak_viewer.py`

---

## Mode natif macOS (sans Docker)

Testé sur Apple Silicon : lerobot, depthai, opencv et torch (avec MPS) s'installent et s'importent nativement, sans adaptation.

```bash
uv sync --no-dev --extra cpu   # équivalent natif de ce que fait le Dockerfile
source .venv/bin/activate      # à refaire dans chaque nouveau terminal
```

⚠️ Pas de `uv pip install -e .` : le projet n'a pas de `[build-system]`, seul `uv sync` fonctionne (sinon erreur de build setuptools sans rapport avec macOS).

⚠️ Sans l'activation (ou `.venv/bin/python` à la place de `python3` ci-dessous), les commandes `python3 scripts/...` de cette section utilisent l'interpréteur système et échouent avec `ModuleNotFoundError: No module named 'lerobot'`.

**Ports série.** macOS n'attribue pas `/dev/ttyACM0`/`ACM1` : chaque bras apparaît en `/dev/cu.usbmodemXXXX`, nom variable, sans ordre stable. Identifier automatiquement quel port est le follower et lequel est le leader (par la tension : follower ~12V, leader ~5V) et obtenir les `export` à copier-coller :

```bash
python3 scripts/robot/identify_arms.py
```

Puis coller les `export` affichés, ou passer les ports explicitement :

```bash
# commandes lerobot (teleop, calibrate...) :
--robot.port=/dev/cu.usbmodemXXXX --teleop.port=/dev/cu.usbmodemYYYY

# scripts/robot/check_devices.py et scan_motors.py :
FOLLOWER_PORT=/dev/cu.usbmodemXXXX LEADER_PORT=/dev/cu.usbmodemYYYY python3 scripts/robot/check_devices.py

# scripts à un seul bras (diag_voltage.py, move_central_position.py...) :
ROBOT_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/diag_voltage.py
```

**`make setup-host`, `make setup-udev`, `make setup-oak` : sans objet sur macOS.** Ce sont des réglages Linux (groupe `dialout`, règles udev). Ils échouent maintenant avec un message explicite au lieu d'une erreur système cryptique : rien à faire à la place, aucune règle système n'est nécessaire sur macOS.

**Caméras.** Webcam USB générique (U20CAM, ET-S231...) : `camera_viewer.py` et les scripts qui réutilisent `V4L2Camera` basculent automatiquement sur le backend caméra par défaut de macOS, rien à changer. OAK-D Lite : fonctionne sans udev, `depthai` détecte le device directement.

⚠️ Au lancement de `lerobot-teleoperate` (et probablement d'autres commandes qui chargent `cv2` et `av`), macOS affiche des messages `objc[...]: Class AVFFrameReceiver is implemented in both ...` qui parlent de « crashes mystérieux ». Sans gravité : `opencv-python` et `av` embarquent chacun leur propre copie de `libavdevice`, d'où le conflit de classes Objective-C signalé. La téléopération tourne normalement derrière (confirmé à 60Hz réels). Rien à corriger de ton côté.

Non testé : Windows/WSL.

### Alternative : VM Linux sur Mac (Vagrant + VMware Fusion)

Marche aussi, testé réellement sur Apple Silicon (Fusion 13.5.2, box `bento/ubuntu-24.04` arm64, provider `vmware_desktop`) : `uv sync`, les trois `make setup-*` (qui fonctionnent normalement, contrairement au mode natif macOS ci-dessus), le fallback caméra en V4L2, et les bras branchés en USB passthrough (check-devices, scan-motors, diag-voltage, calibration, téléopération).

Deux points pratiques :
- Le passthrough USB n'est pas scriptable simplement : il passe par le menu Fusion **Virtual Machine > USB & Bluetooth** (VM démarrée avec fenêtre, pas en mode headless/`nogui`), à faire une fois par device avant de rebrancher.
- Les fichiers de calibration vivent sur le disque de la VM, pas sur celui du Mac hôte : pas besoin de recalibrer si déjà fait côté macOS natif, un `scp` des deux `.json` (`~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/*.json`) vers la VM suffit, mêmes valeurs.

---

## Features

Téléopération leader/follower, enregistrement de datasets pour l'apprentissage par imitation, et rejeu d'un épisode enregistré.

Côté entraînement : policies ACT / SmolVLA / π0, sur GPU local ou distant. L'évaluation peut tourner en local (robot + policy sur la même machine) ou à distance, avec la policy via serveur d'inférence pendant que robot et caméra restent en local.

---

## Commandes

Toutes les commandes sont `make <cible>`. Le Makefile racine inclut `mk/{setup,robot,dataset,camera}.mk`, la cible se trouve dans le fichier correspondant à son domaine.

### Setup / image

| Commande | Description |
|----------|-------------|
| `make build` | Build l'image Docker `robotkraft:latest` |
| `make shell` | Shell dans `lerobot-base` (sans robot) |
| `make lock` | Régénère `uv.lock` |
| `make setup-host` | Ajoute l'utilisateur au groupe `dialout` |
| `make setup-udev` | Crée `/dev/lerobot_follower` + `/dev/lerobot_leader` |
| `make setup-oak` | Crée la règle udev OAK-D Lite (`/dev/oak`) |

### Téléopération / calibration / diagnostic

| Commande | Description |
|----------|-------------|
| `make teleop` | Téléopération leader/follower en direct |
| `make calibrate-follower` / `make calibrate-leader` | Calibration d'un bras (une fois) |
| `make check-devices` | Vérifie `ACM0`=follower, `ACM1`=leader |
| `make scan-motors` | Liste les IDs des servo-moteurs |
| `make check-voltage` | Tension des servos (follower + leader) |
| `make check-voltage-follower` / `make check-voltage-leader` | Tension d'un seul bras |
| `make check-oak` | Vérifie que la caméra OAK-D est bien détectée par l'hôte |
| `make script-follower FILE=...` / `make script-leader FILE=...` | Lance un script Python avec accès direct à un seul bras |

### Enregistrement / replay

| Commande | Description |
|----------|-------------|
| `make record HF_USER=... TASK=...` | Enregistre un dataset de téléopération (`[NUM_EPISODES]`, `[EPISODE_TIME]`, `[RESET_TIME]`, `[RESUME]`) |
| `make replay-episode HF_USER=... TASK=... [EPISODE=0]` | Rejoue un épisode d'un dataset enregistré (follower seul) |

### VLA / ordres textuels (SmolVLA / π0)

Piloter le follower par un **ordre textuel** (ex : « pick the red cube and place it in the rack ») via une VLA fine-tunée, au lieu d'une policy d'imitation ACT. Les caméras sont nommées `camera1` (scène OAK-D) + `camera2` (poignet UVC) pour correspondre au checkpoint `lerobot/smolvla_base` ; le chemin d'inférence asynchrone ne transmet aucun `rename_map`, ce nommage est donc obligatoire.

| Commande | Description |
|----------|-------------|
| `make record-vla HF_USER=... TASK=... LANG_TASK="..." [WRIST_DEV=0]` | Enregistre un dataset VLA (2 caméras + instruction anglaise) par téléopération |
| `make train-smolvla HF_USER=... TASK=... [STEPS=2000]` | Fine-tune une SmolVLA depuis `lerobot/smolvla_base` sur le dataset (GPU) |
| `make train-pi0 HF_USER=... TASK=... [STEPS=2000]` | Fine-tune un π0 (optionnel, plus gourmand ; PEFT/LoRA conseillé sur slice A100 20 Go) |
| `make push-checkpoint-vla HF_USER=... TASK=...` | Push le checkpoint SmolVLA sur HF Hub (requis avant `eval-vla` distante) |
| `make eval-vla TASK=... LANG_TASK="..." HF_USER=... SERVER=ip:port [POLICY_TYPE=smolvla]` | Évalue la VLA : policy sur GPU distant, robot + caméras en local |

> `LANG_TASK` est l'**entrée langage** de la VLA (instruction en anglais) : stockée dans le dataset à l'enregistrement, puis envoyée au policy serveur frame à frame à l'inférence. C'est elle qui pilote le bras. La variante π0 a des caméras nommées `camera0/1/2` : pour un dataset dédié π0, renommer les caméras dans `scripts/shell/record_vla_with_oak.sh` (voir le commentaire de `train-pi0` dans `mk/vla.mk`).

### Entraînement / évaluation

| Commande | Description |
|----------|-------------|
| `make train HF_USER=... TASK=...` | Entraîne une policy ACT (GPU local) |
| `make eval TASK=... [HF_USER=...] [CHECKPOINT=last]` | Évalue une policy sur le robot réel, inférence locale |
| `make push-checkpoint HF_USER=... TASK=...` | Push un checkpoint entraîné sur HF Hub (requis avant `eval-remote`) |
| `make policy-server` | Lance le serveur d'inférence en local |
| `make eval-remote HF_USER=... TASK=... SERVER=ip:port` | Évalue avec policy sur GPU distant, robot + caméra en local |

### Vision / caméra

| Commande | Description |
|----------|-------------|
| `make detect-cameras` | Détection des caméras USB disponibles (U20CAM, ET-S231 et autres UVC) |
| `make detect-oak` | Détection caméra OAK-D Lite (option) |
| `make view-oak` | Preview live OAK-D Lite (fenêtre X11), `Q`/Échap pour quitter |
| `make calibrate-wb` | Calibration de la balance des blancs |
| `make view-camera DEVICE=/dev/video2` | Preview live webcam USB (U20CAM, ET-S231 et autres UVC), exécuté sur l'hôte |
| `make photo [FILE=...] [CROP_X/Y/W/H=...]` | Capture une photo |

---

## Bloqué ?

Pendant l'événement, le canal d'entraide et le contact de l'organisation vous sont
communiqués séparément.

---

*The French content above is the reference. English content below.*

# RobotKraft, participant edition

> Technical base for the RobotKraft challenge: ready-to-use Docker environment, arm and camera
> detection, hardware diagnostics, and shortcuts to the LeRobot commands (teleoperation, dataset
> recording, training, local or remote-GPU evaluation).
>
> Perception and strategy are not provided: that's what the challenge evaluates, it's up to you
> to write them.
>
> Information specific to each edition (venue, schedule, rules, challenges, help channel) is
> shared separately.

Space/chemistry robotics for an AI & Robotics hackathon: SO-ARM101 arms (leader/follower), imitation learning with [LeRobot](https://huggingface.co/docs/lerobot), all containerized under Docker.

---

## Safety: plugging order

⚠️ Two mistakes destroy hardware.

**Power first, USB second.** If USB is plugged in before power, the USB port ends up powering the board and can burn out. Order: power, then follower USB, then leader USB. To unplug, the exact reverse.

**Each arm has its own voltage, and the connectors are interchangeable.** Follower 12V, leader 5V. Putting 12V into a leader destroys it. Color coding: green = 5V = leader, white = 12V = follower. Green wire to green wire, white wire to white wire.

- `input voltage error` under load (~5.4V): insufficient power.
- `/dev/ttyACM0`/`ACM1` ports are assigned by plugging order (follower first = `ACM0`). Check `ls /dev/ttyACM*` before each session.
- Both arms share the same USB VID:PID (WCH chip), so that won't tell them apart: if in doubt about which is which, trust the voltage (follower 12V, leader 5V), not the serial number or plugging order. `python3 scripts/robot/identify_arms.py` automates this (Linux and macOS).

---

## Architecture

| Component               |
|-------------------------|
| SO-ARM101               |
| LeRobot (HuggingFace)   |
| Wrist camera: InnoMaker U20CAM or ET-S231 depending on the station (both USB UVC) |
| Scene camera (optional, 11 available on site): OAK-D Lite (depthai) |
| Python 3.12.9 + uv      |
| Docker                  |

---

## Docker services

| Service |
|---------|
| `lerobot-base` |
| `lerobot` |
| `lerobot-gpu` |
| `lerobot-follower` |
| `lerobot-leader` |
| `lerobot-camera` |

> `lerobot` requires both `/dev/lerobot_follower` **and** `/dev/lerobot_leader`. Only one arm plugged in: use `lerobot-follower` / `lerobot-leader` (they mount `/dev/ttyACM0`/`ACM1`, no udev symlink).

---

## Prerequisites

- [Docker](https://www.docker.com/)
- `HF_TOKEN` in `.env` (see `.env.example`) to push/pull HuggingFace datasets and checkpoints (get one via Access Token > Create New Token on the HuggingFace site)

---

## Installation

The image is ready, there's nothing to build. Pull it rather than building it, you'll save
about twenty minutes and the venue's bandwidth:

```bash
docker compose pull      # pulls the image from ghcr.io
```

**Do this at home before coming.** The image weighs about 2.7 GB, and dozens of simultaneous
downloads on the venue's connection is not a good idea.

If your laptop has an NVIDIA GPU and you want to train locally, switch to the CUDA
variant (about 9 GB):

```bash
make use-cuda            # writes the choice to .env, so it persists
docker compose pull
make which-image         # check: should print GPU vu par torch : True
```

Otherwise, training happens on the remote GPU servers, and the default variant is enough.
`make use-cpu` switches back.

⚠️ Don't pass `ROBOTKRAFT_IMAGE=...` directly in front of a command: the value only applies
to that one command, and `make train` would silently fall back to the CPU variant, training
on the processor without warning.

### Set up your machine, required even if you pull the image

These three commands configure your system, not the image. Without them, the arms and the
camera won't be visible from the containers.

```bash
make setup-host # dialout group (log out and back in right after)
make setup-udev # udev rules and /dev/lerobot_follower / _leader symlinks
make setup-oak  # udev rule for the OAK-D Lite camera (once)
```

Also copy the configuration file:

```bash
cp .env.example .env    # then fill in HF_TOKEN
```

### Rebuild the image, only if you change the Dockerfile or the dependencies

```bash
make build
```

### ⛔ Don't reconfigure the servo motors

The servo IDs are **already set on the arms you're being lent**, and the whole chain depends
on them. Redefining them breaks the arm for you and for the next team using it, and it then
has to be reconfigured servo by servo.

**Never run `lerobot-setup-motors`, or any motor ID configuration command.** If a servo seems
silent or mis-numbered, it's not for you to fix: go find a coach.

To check that the servos respond, without changing anything:

```bash
make scan-motors    # lists the IDs seen on the bus
make check-voltage  # voltage of each servo
```

### Calibrating the robots

```bash
make calibrate-follower # Follower calibration (once)
make calibrate-leader   # Leader calibration (once)
```

> Calibrations are stored in `.cache/` (Docker volume). To recalibrate: delete the matching `.json` file in `.cache/huggingface/lerobot/calibration/`.

### Post-installation check

```bash
ls /dev/ttyACM*    # follower/leader detected
make check-devices # checks ACM0=follower, ACM1=leader
make check-voltage # voltage of each servo (follower + leader)
```

### Wrist camera: InnoMaker U20CAM or ET-S231 depending on the station

- Standard USB UVC camera (U20CAM on kits 1-8, ET-S231 built in on kits 9-11), no udev rule needed
- `make detect-cameras` -> lists the USB cameras available in the container
- `make view-camera DEVICE=/dev/videoX` -> live preview on the host, to find the right device
- ET-S231 (kits 9-11): MANUAL focus, via the lens ring. Before recording demonstrations, check in a camera preview (`make view-camera DEVICE=/dev/videoX` on Linux, or the system's camera app on macOS/Windows) that the image is sharp at the distance where the arm picks up objects, then don't touch the ring again: a setting that drifts mid-way makes your training data inconsistent.

### Scene camera (optional, 11 available on site): OAK-D Lite

- **USB3 required**
- `make setup-oak` once (udev rule + `/dev/oak` symlink)
- `make detect-oak` -> should capture an RGB frame to confirm the camera is detected
- `make view-oak [WIDTH=640 HEIGHT=480 FPS=30]` -> live preview (X11 window), `Q`/Esc to quit
- **Never** `import depthai` from the host, **nor** `docker run` (always `docker compose run lerobot-camera ...`): an ephemeral container can leave the device locked
- Native mode (no Docker, see dedicated section): two terminals, `WIDTH=640 HEIGHT=480 python3 scripts/camera/oak_zmq_server.py` then `python3 scripts/camera/oak_viewer.py`

---

## Native macOS mode (no Docker)

Tested on Apple Silicon: lerobot, depthai, opencv and torch (with MPS) install and import natively, no adaptation needed.

```bash
uv sync --no-dev --extra cpu   # native equivalent of what the Dockerfile does
source .venv/bin/activate      # redo this in every new terminal
```

⚠️ No `uv pip install -e .`: the project has no `[build-system]`, only `uv sync` works (otherwise a setuptools build error unrelated to macOS).

⚠️ Without activation (or using `.venv/bin/python` instead of `python3` below), the `python3 scripts/...` commands in this section use the system interpreter and fail with `ModuleNotFoundError: No module named 'lerobot'`.

**Serial ports.** macOS doesn't assign `/dev/ttyACM0`/`ACM1`: each arm shows up as `/dev/cu.usbmodemXXXX`, name varies, no stable order. Automatically identify which port is the follower and which is the leader (by voltage: follower ~12V, leader ~5V) and get copy-paste-ready `export` lines:

```bash
python3 scripts/robot/identify_arms.py
```

Then paste the `export` lines it prints, or pass the ports explicitly:

```bash
# lerobot commands (teleop, calibrate...):
--robot.port=/dev/cu.usbmodemXXXX --teleop.port=/dev/cu.usbmodemYYYY

# scripts/robot/check_devices.py and scan_motors.py:
FOLLOWER_PORT=/dev/cu.usbmodemXXXX LEADER_PORT=/dev/cu.usbmodemYYYY python3 scripts/robot/check_devices.py

# single-arm scripts (diag_voltage.py, move_central_position.py...):
ROBOT_PORT=/dev/cu.usbmodemXXXX python3 scripts/robot/diag_voltage.py
```

**`make setup-host`, `make setup-udev`, `make setup-oak`: not applicable on macOS.** These are Linux settings (`dialout` group, udev rules). They now fail with an explicit message instead of a cryptic system error: there's nothing to do instead, no system rule is needed on macOS.

**Cameras.** Generic USB webcam (U20CAM, ET-S231...): `camera_viewer.py` and scripts reusing `V4L2Camera` automatically fall back to macOS's default camera backend, nothing to change. OAK-D Lite: works without udev, `depthai` detects the device directly.

⚠️ When starting `lerobot-teleoperate` (and probably other commands that load both `cv2` and `av`), macOS prints `objc[...]: Class AVFFrameReceiver is implemented in both ...` messages warning about "mysterious crashes." Harmless: `opencv-python` and `av` each bundle their own copy of `libavdevice`, hence the Objective-C class conflict. Teleoperation runs fine behind it (confirmed at a real 60Hz). Nothing to fix on your end.

Not tested: Windows/WSL.

### Alternative: Linux VM on a Mac (Vagrant + VMware Fusion)

Also works, tested for real on Apple Silicon (Fusion 13.5.2, `bento/ubuntu-24.04` arm64 box, `vmware_desktop` provider): `uv sync`, all three `make setup-*` targets (which work normally here, unlike native macOS above), the V4L2 camera fallback, and the arms over USB passthrough (check-devices, scan-motors, diag-voltage, calibration, teleoperation).

Two practical points:
- USB passthrough isn't easily scriptable: it goes through Fusion's **Virtual Machine > USB & Bluetooth** menu (VM started with a window, not headless/`nogui`), once per device before reconnecting.
- Calibration files live on the VM's disk, not the host Mac's: no need to recalibrate if already done on native macOS, `scp`-ing the two `.json` files (`~/.cache/huggingface/lerobot/calibration/{robots/so_follower,teleoperators/so_leader}/*.json`) to the VM is enough, same values.

---

## Features

Leader/follower teleoperation, dataset recording for imitation learning, and replay of a recorded episode.

On the training side: ACT / SmolVLA / π0 policies, on local or remote GPU. Evaluation can run locally (robot + policy on the same machine) or remotely, with the policy served over an inference server while the robot and camera stay local.

---

## Commands

All commands are `make <target>`. The root Makefile includes `mk/{setup,robot,dataset,camera}.mk`, each target lives in the file matching its domain.

### Setup / image

| Command | Description |
|----------|-------------|
| `make build` | Builds the `robotkraft:latest` Docker image |
| `make shell` | Shell into `lerobot-base` (no robot) |
| `make lock` | Regenerates `uv.lock` |
| `make setup-host` | Adds the user to the `dialout` group |
| `make setup-udev` | Creates `/dev/lerobot_follower` + `/dev/lerobot_leader` |
| `make setup-oak` | Creates the OAK-D Lite udev rule (`/dev/oak`) |

### Teleoperation / calibration / diagnostics

| Command | Description |
|----------|-------------|
| `make teleop` | Live leader/follower teleoperation |
| `make calibrate-follower` / `make calibrate-leader` | Calibrates one arm (once) |
| `make check-devices` | Checks `ACM0`=follower, `ACM1`=leader |
| `make scan-motors` | Lists the servo motor IDs |
| `make check-voltage` | Voltage of the servos (follower + leader) |
| `make check-voltage-follower` / `make check-voltage-leader` | Voltage of a single arm |
| `make check-oak` | Checks that the OAK-D camera is detected by the host |
| `make script-follower FILE=...` / `make script-leader FILE=...` | Runs a Python script with direct access to a single arm |

### Recording / replay

| Command | Description |
|----------|-------------|
| `make record HF_USER=... TASK=...` | Records a teleoperation dataset (`[NUM_EPISODES]`, `[EPISODE_TIME]`, `[RESET_TIME]`, `[RESUME]`) |
| `make replay-episode HF_USER=... TASK=... [EPISODE=0]` | Replays an episode from a recorded dataset (follower only) |

### VLA / text commands (SmolVLA / π0)

Drive the follower with a **text instruction** (e.g. "pick the red cube and place it in the rack") via a fine-tuned VLA, instead of an ACT imitation policy. Cameras are named `camera1` (OAK-D scene) + `camera2` (wrist UVC) to match the `lerobot/smolvla_base` checkpoint; the async inference path transmits no `rename_map`, so this naming is mandatory.

| Command | Description |
|----------|-------------|
| `make record-vla HF_USER=... TASK=... LANG_TASK="..." [WRIST_DEV=0]` | Records a VLA dataset (2 cameras + English instruction) via teleoperation |
| `make train-smolvla HF_USER=... TASK=... [STEPS=2000]` | Fine-tunes a SmolVLA from `lerobot/smolvla_base` on the dataset (GPU) |
| `make train-pi0 HF_USER=... TASK=... [STEPS=2000]` | Fine-tunes π0 (optional, heavier; PEFT/LoRA recommended on the 20 GB A100 slice) |
| `make push-checkpoint-vla HF_USER=... TASK=...` | Pushes the SmolVLA checkpoint to the HF Hub (required before remote `eval-vla`) |
| `make eval-vla TASK=... LANG_TASK="..." HF_USER=... SERVER=ip:port [POLICY_TYPE=smolvla]` | Evaluates the VLA: policy on a remote GPU, robot + cameras local |

> `LANG_TASK` is the VLA's **language input** (English instruction): stored in the dataset at recording time, then sent to the policy server frame by frame at inference. It is what drives the arm. The π0 variant expects cameras named `camera0/1/2`: for a dedicated π0 dataset, rename the cameras in `scripts/shell/record_vla_with_oak.sh` (see the `train-pi0` comment in `mk/vla.mk`).

### Training / evaluation

| Command | Description |
|----------|-------------|
| `make train HF_USER=... TASK=...` | Trains an ACT policy (local GPU) |
| `make eval TASK=... [HF_USER=...] [CHECKPOINT=last]` | Evaluates a policy on the real robot, local inference |
| `make push-checkpoint HF_USER=... TASK=...` | Pushes a trained checkpoint to the HF Hub (required before `eval-remote`) |
| `make policy-server` | Starts the local inference server |
| `make eval-remote HF_USER=... TASK=... SERVER=ip:port` | Evaluates with the policy on a remote GPU, robot + camera local |

### Vision / camera

| Command | Description |
|----------|-------------|
| `make detect-cameras` | Detects the available USB cameras (U20CAM, ET-S231 and other UVC) |
| `make detect-oak` | Detects the OAK-D Lite camera (optional) |
| `make view-oak` | Live OAK-D Lite preview (X11 window), `Q`/Esc to quit |
| `make calibrate-wb` | White balance calibration |
| `make view-camera DEVICE=/dev/video2` | Live preview of a USB webcam (U20CAM, ET-S231 and other UVC), runs on the host |
| `make photo [FILE=...] [CROP_X/Y/W/H=...]` | Takes a photo |

---

## Stuck?

During the event, the help channel and the organization's contact are shared separately.
