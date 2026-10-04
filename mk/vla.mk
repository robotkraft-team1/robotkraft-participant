# Architecture VLA (SmolVLA + pi0 optionnel)
# Piloter le follower par ordre textuel via une VLA fine-tune.
#
# La "task" (chaine textuelle) est l'entree langage de la VLA : stockee dans le
# dataset via --dataset.single_task, puis envoyee frame a frame au policy_server
# par le robot_client au moment de l'inference (--task=).
#
# Les cameras sont nommees camera1 (scene OAK-D) / camera2 (poignet UVC) pour
# correspondre aux input_features du checkpoint lerobot/smolvla_base. Le chemin
# d'inference asynchrone ne transmet AUCUN rename_map, donc ce nommage est
# obligatoire pour que le modele reconnaisse les images (train + inference).

# Enregistre un dataset VLA (2 cameras + instruction anglaise) par teleoperation.
# Usage : make record-vla HF_USER=user TASK=task LANG_TASK="pick the red cube" [WRIST_DEV=0]
record-vla:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make record-vla HF_USER=monuser TASK=ma_tache LANG_TASK="pick the red cube"))
	@$(if $(TASK),,$(error TASK requis - Usage: make record-vla HF_USER=monuser TASK=ma_tache LANG_TASK="..."))
	@$(if $(LANG_TASK),,$(error LANG_TASK requis (instruction anglaise) - ex: LANG_TASK="pick the red cube and place it in the rack"))
	@xhost +local:docker >/dev/null 2>&1 || true
	docker compose run --rm \
		-e HF_USER=$(HF_USER) \
		-e TASK=$(TASK) \
		-e "LANG_TASK=$(LANG_TASK)" \
		-e NUM_EPISODES=$(or $(NUM_EPISODES),10) \
		-e EPISODE_TIME=$(or $(EPISODE_TIME),10) \
		-e RESET_TIME=$(or $(RESET_TIME),10) \
		-e WRIST_DEV=$(or $(WRIST_DEV),0) \
		-e SCENE_WIDTH=$(or $(SCENE_WIDTH),1280) \
		-e SCENE_HEIGHT=$(or $(SCENE_HEIGHT),720) \
		-e VCODEC=$(or $(VCODEC),h264) \
		-e PUSH_TO_HUB=$(or $(PUSH_TO_HUB),false) \
		$(if $(RESUME),-e RESUME=$(RESUME)) \
		lerobot bash scripts/shell/record_vla_with_oak.sh

# Fine-tune une VLA SmolVLA depuis lerobot/smolvla_base sur le dataset VLA.
# Les flags freeze_vision_encoder / train_expert_only sont les presets par defaut
# de SmolVLA (seul l'expert action + la projection state sont entrainees) -> leger
# et rapide, tient sur une slice A100 20 Go.
# Usage : make train-smolvla HF_USER=user TASK=task [STEPS=2000] [BATCH=4]
train-smolvla:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make train-smolvla HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK requis - Usage: make train-smolvla HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-gpu \
		lerobot-train \
		--policy.type=smolvla \
		--policy.pretrained_path=lerobot/smolvla_base \
		--policy.device=cuda \
		--policy.push_to_hub=false \
		--dataset.repo_id=$(HF_USER)/$(TASK) \
		--batch_size=$(or $(BATCH),4) \
		--num_workers=4 \
		--steps=$(or $(STEPS),2000) \
		--save_freq=$(or $(SAVE_FREQ),1000) \
		--output_dir=outputs/smolvla_$(TASK)

# Fine-tune une VLA pi0 (optionnel, plus puissant mais plus gourmand).
# NB : lerobot/pi0 est pre-entraine avec des cameras nommees camera0/1/2, pas
# camera1/2. Pour un dataset dedie pi0, renommer les cameras dans record_vla_
# with_oak.sh en camera0 (scene) / camera1 (poignet), puis ajouter un empty
# camera via --policy.empty_cameras si necessaire. Sur slice A100 20 Go, le full
# fine-tune depasse : preferrer PEFT/LoRA (--policy.use_peft=true --peft.r=16,
# avec --peft.target_modules a preciser). Sur un GPU complet (Agora), le full
# fine-tune passe avec --policy.gradient_checkpointing=true.
# Usage : make train-pi0 HF_USER=user TASK=task [STEPS=2000]
train-pi0:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make train-pi0 HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK requis - Usage: make train-pi0 HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-gpu \
		lerobot-train \
		--policy.type=pi0 \
		--policy.pretrained_path=lerobot/pi0 \
		--policy.device=cuda \
		--policy.dtype=bfloat16 \
		--policy.use_peft=$(or $(USE_PEFT),false) \
		--policy.gradient_checkpointing=true \
		--policy.push_to_hub=false \
		--dataset.repo_id=$(HF_USER)/$(TASK) \
		--batch_size=$(or $(BATCH),2) \
		--num_workers=4 \
		--steps=$(or $(STEPS),2000) \
		--save_freq=$(or $(SAVE_FREQ),1000) \
		--output_dir=outputs/pi0_$(TASK) \
		$(if $(USE_PEFT),--peft.r=$(or $(PEFT_R),16))

# Push le checkpoint VLA (smolvla) sur HF Hub, requis avant eval-vla distante.
# Usage : make push-checkpoint-vla HF_USER=monuser TASK=task [CHECKPOINT=last]
push-checkpoint-vla:
	@$(if $(HF_USER),,$(error HF_USER requis - Usage: make push-checkpoint-vla HF_USER=monuser TASK=ma_tache))
	@$(if $(TASK),,$(error TASK requis - Usage: make push-checkpoint-vla HF_USER=monuser TASK=ma_tache))
	docker compose run --rm lerobot-base \
		hf upload $(HF_USER)/$(TASK) \
		outputs/smolvla_$(TASK)/checkpoints/$(or $(CHECKPOINT),last)/pretrained_model .

# Eval distante d'une VLA : robot + cameras en local, policy sur GPU distant.
# PREREQUIS : make policy-server tourne cote GPU, checkpoint push (ou accessible).
# Convention (comme record-vla) : TASK = nom du dataset, LANG_TASK = l'ordre
# textuel en anglais envoye a la VLA (si absent, il vaut par defaut TASK).
# Usage : make eval-vla TASK=tri_cubes LANG_TASK="pick the red cube" HF_USER=monuser SERVER=ip:port [POLICY_TYPE=smolvla]
eval-vla:
	@$(if $(TASK),,$(error TASK requis (nom du dataset) - ex: make eval-vla TASK=tri_cubes LANG_TASK="pick the red cube" SERVER=ip:port))
	@$(if $(SERVER),,$(error SERVER requis - ex: SERVER=ip:port du policy_server))
	docker compose run --rm \
		-e TASK=$(TASK) \
		-e "LANG_TASK=$(or $(LANG_TASK),$(TASK))" \
		-e SERVER=$(SERVER) \
		$(if $(PRETRAINED_PATH),-e PRETRAINED_PATH=$(PRETRAINED_PATH)) \
		-e HF_USER=$(or $(HF_USER),) \
		-e CHECKPOINT=$(or $(CHECKPOINT),last) \
		-e POLICY_TYPE=$(or $(POLICY_TYPE),smolvla) \
		-e ACTIONS_PER_CHUNK=$(or $(ACTIONS_PER_CHUNK),100) \
		-e CHUNK_SIZE_THRESHOLD=$(or $(CHUNK_SIZE_THRESHOLD),0) \
		-e DEBUG_QUEUE=$(or $(DEBUG_QUEUE),false) \
		-e WRIST_DEV=$(or $(WRIST_DEV),0) \
		-e SCENE_WIDTH=$(or $(SCENE_WIDTH),1280) \
		-e SCENE_HEIGHT=$(or $(SCENE_HEIGHT),720) \
		lerobot bash scripts/shell/eval_vla_with_oak.sh
