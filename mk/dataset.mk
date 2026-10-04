# Enregistre un dataset de téléopération
# Usage : make record HF_USER=user TASK=task_name [WITH_WRIST=true] [WRIST_DEVICE=/dev/video2] [NUM_EPISODES=10] [EPISODE_TIME=10] [RESET_TIME=10] [RESUME=true] [WIDTH=640] [HEIGHT=480] [FPS=30]
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
		-e WITH_WRIST=$(or $(WITH_WRIST),false) \
		$(if $(WRIST_DEVICE),-e WRIST_DEVICE=$(WRIST_DEVICE)) \
		$(if $(RESUME),-e RESUME=$(RESUME)) \
		$(if $(WIDTH),-e WIDTH=$(WIDTH)) \
		$(if $(HEIGHT),-e HEIGHT=$(HEIGHT)) \
		$(if $(FPS),-e FPS=$(FPS)) \
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
# Usage : make train HF_USER=user TASK=task_name [BATCH_SIZE=2] [STEPS=20000] [SAVE_FREQ=5000]
# PREREQUIS : caméra branchée lors du record, ACT requiert observation.image
train:
	@$(if $(HF_USER),,$(error HF_USER non défini - Usage: make train HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK non défini - Usage: make train HF_USER=monuser TASK=ma_tache))
	docker compose run --rm \
		-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
		lerobot-gpu \
		lerobot-train \
		--policy.type=act \
		--dataset.repo_id=$(HF_USER)/$(TASK) \
		--policy.push_to_hub=false \
		--output_dir=outputs/$(TASK) \
		--batch_size=$(or $(BATCH_SIZE),2) \
		--policy.use_amp=$(or $(USE_AMP),true) \
		$(if $(STEPS),--steps=$(STEPS)) \
		$(if $(SAVE_FREQ),--save_freq=$(SAVE_FREQ))

# Évalue le modèle entraîné sur le robot réel (inférence policy réelle, via lerobot-record)
# Usage : make eval TASK=task_name [CHECKPOINT=last] [HF_USER=monuser] [WITH_WRIST=true] [WRIST_DEVICE=/dev/video2] [NUM_EPISODES=5] [EPISODE_TIME=30] [WIDTH=640] [HEIGHT=480] [FPS=30]
# PREREQUIS : caméra OAK branchée
eval:
	@$(if $(TASK),,$(error TASK non défini - Usage: make eval TASK=ma_tache))
	docker compose run --rm \
		-e POLICY_PATH=$(if $(HF_USER),$(HF_USER)/$(TASK),outputs/$(TASK)/checkpoints/$(or $(CHECKPOINT),last)/pretrained_model) \
		-e TASK=$(TASK) \
		-e NUM_EPISODES=$(or $(NUM_EPISODES),5) \
		-e EPISODE_TIME=$(or $(EPISODE_TIME),30) \
		-e RESET_TIME=$(or $(RESET_TIME),10) \
		-e WITH_WRIST=$(or $(WITH_WRIST),false) \
		$(if $(WRIST_DEVICE),-e WRIST_DEVICE=$(WRIST_DEVICE)) \
		$(if $(WIDTH),-e WIDTH=$(WIDTH)) \
		$(if $(HEIGHT),-e HEIGHT=$(HEIGHT)) \
		$(if $(FPS),-e FPS=$(FPS)) \
		lerobot bash scripts/shell/eval_with_oak.sh

# Push un dataset enregistré sur HF hub (pour entraînement sur serveur distant)
# Usage : make push-dataset HF_USER=monuser TASK=task_name
push-dataset:
	@$(if $(HF_USER),,$(error HF_USER non défini - Usage: make push-dataset HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK non défini - Usage: make push-dataset HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-base \
		hf upload --repo-type dataset $(HF_USER)/$(TASK) \
		.cache/huggingface/lerobot/$(HF_USER)/$(TASK) .

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
# Usage : make eval-remote HF_USER=monuser TASK=task_name SERVER=ip:port [ACTIONS_PER_CHUNK=50] [WIDTH=640] [HEIGHT=480] [FPS=30]
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
		$(if $(WIDTH),-e WIDTH=$(WIDTH)) \
		$(if $(HEIGHT),-e HEIGHT=$(HEIGHT)) \
		$(if $(FPS),-e FPS=$(FPS)) \
		lerobot bash scripts/shell/eval_remote_with_oak.sh
