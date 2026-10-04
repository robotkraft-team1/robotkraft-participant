# Enregistre un dataset de téléopération
# Usage : make record HF_USER=user TASK=task_name [NUM_EPISODES=10] [EPISODE_TIME=10] [RESET_TIME=10] [RESUME=true]
# Si dataset existant (run échoué) renommer ou supprimer le dataset existant
record:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make record HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK requis - Usage: make record HF_USER=monuser TASK=ma_tache))
	@xhost +local:docker >/dev/null 2>&1 || true
	docker compose run --rm \
		-e HF_USER=$(HF_USER) \
		-e TASK=$(TASK) \
		-e NUM_EPISODES=$(or $(NUM_EPISODES),10) \
		-e EPISODE_TIME=$(or $(EPISODE_TIME),10) \
		-e RESET_TIME=$(or $(RESET_TIME),10) \
		-e PUSH_TO_HUB=$(or $(PUSH_TO_HUB),false) \
		$(if $(WRIST_DEVICE),-e WRIST_DEVICE=$(WRIST_DEVICE)) \
		$(if $(WRIST_WIDTH),-e WRIST_WIDTH=$(WRIST_WIDTH)) \
		$(if $(WRIST_HEIGHT),-e WRIST_HEIGHT=$(WRIST_HEIGHT)) \
		$(if $(RESUME),-e RESUME=$(RESUME)) \
		lerobot bash scripts/shell/record_with_oak.sh

# Rejoue un épisode d'un dataset enregistré via make record (follower seul)
# Usage : make replay-episode HF_USER=user TASK=task_name [EPISODE=0]
replay-episode:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make replay-episode HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK requis - Usage: make replay-episode HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-follower \
		lerobot-replay \
		--robot.type=so101_follower \
		--robot.port=/dev/ttyACM0 \
		--robot.id=follower_arm \
		--dataset.repo_id=$(HF_USER)/$(TASK) \
		--dataset.episode=$(or $(EPISODE),0) \
		--play_sounds=false

# Entraîne un modèle ACT sur le dataset enregistré
# Usage : make train HF_USER=user TASK=task_name
# PREREQUIS : caméra branchée lors du record, ACT requiert observation.image
train:
	@$(if $(HF_USER),,$(error HF_USER non défini - Usage: make train HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK non défini - Usage: make train HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-gpu \
		lerobot-train \
		--policy.type=act \
		--dataset.repo_id=$(HF_USER)/$(TASK) \
		--policy.push_to_hub=false \
		--output_dir=outputs/$(TASK)

# Évalue le modèle entraîné sur le robot réel (inférence policy réelle, via lerobot-record)
# Sans HF_USER : pointe sur le modèle final local (outputs/$(TASK)/pretrained_model)
# Usage : make eval TASK=task_name [HF_USER=monuser] [NUM_EPISODES=5] [EPISODE_TIME=30]
# PREREQUIS : caméra OAK branchée
eval:
	@$(if $(TASK),,$(error TASK non défini - Usage: make eval TASK=ma_tache))
	@echo "Nettoyage du dataset d'éval résiduel local/eval_$(TASK)..."
	docker compose run --rm lerobot-base python3 -c "import shutil; shutil.rmtree('/workspace/.cache/huggingface/lerobot/local/eval_$(TASK)', ignore_errors=True)"
	docker compose run --rm \
		-e POLICY_PATH=$(if $(HF_USER),$(HF_USER)/$(TASK),outputs/$(TASK)/pretrained_model) \
		-e TASK=$(TASK) \
		-e NUM_EPISODES=$(or $(NUM_EPISODES),5) \
		-e EPISODE_TIME=$(or $(EPISODE_TIME),30) \
		-e RESET_TIME=$(or $(RESET_TIME),10) \
		lerobot bash scripts/shell/eval_with_oak.sh

# Push un checkpoint entraîné sur HF hub (nécessaire pour eval-remote)
# Usage : make push-checkpoint HF_USER=monuser TASK=task_name [CHECKPOINT=last]
push-checkpoint:
	@$(if $(HF_USER),,$(error HF_USER non défini - Usage: make push-checkpoint HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK non défini - Usage: make push-checkpoint HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-base \
		hf upload $(HF_USER)/$(TASK) \
		outputs/$(TASK)/checkpoints/$(or $(CHECKPOINT),last)/pretrained_model .

# Lance le serveur d'inférence policy sur GPU. A exécuter sur la machine distante
# Usage : make policy-server
policy-server:
	docker compose run --rm -p 8080:8080 lerobot-gpu \
		python -m lerobot.async_inference.policy_server \
		--host=0.0.0.0 --port=8080 --fps=30

# Eval réelle avec inférence GPU distante (robot + caméra restent branchés ici, policy tourne sur hôte distant)
# PREREQUIS : make push-checkpoint fait avant, make policy-server tourne côté hôte distant
# Usage : make eval-remote HF_USER=monuser TASK=task_name SERVER=ip:port [ACTIONS_PER_CHUNK=50]
eval-remote:
	@$(if $(HF_USER),,$(error HF_USER non défini - Usage: make eval-remote HF_USER=monuser TASK=ma_tache SERVER=ip:port))
	@$(if $(TASK),,$(error TASK non défini - Usage: make eval-remote HF_USER=monuser TASK=ma_tache SERVER=ip:port))
	@$(if $(SERVER),,$(error SERVER non défini - Usage: make eval-remote HF_USER=monuser TASK=ma_tache SERVER=ip:port))
	docker compose run --rm \
		-e TASK=$(TASK) \
		-e SERVER_ADDRESS=$(SERVER) \
		-e PRETRAINED_PATH=$(HF_USER)/$(TASK) \
		-e ACTIONS_PER_CHUNK=$(or $(ACTIONS_PER_CHUNK),100) \
		-e CHUNK_SIZE_THRESHOLD=$(or $(CHUNK_SIZE_THRESHOLD),0) \
		-e DEBUG_QUEUE=$(or $(DEBUG_QUEUE),false) \
		lerobot bash scripts/shell/eval_remote_with_oak.sh
