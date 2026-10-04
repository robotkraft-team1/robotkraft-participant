.PHONY: build lock shell shell-robot up down teleop record replay-episode train eval push-checkpoint policy-server eval-remote setup-host setup-udev setup-oak check-devices calibrate-follower calibrate-leader scan-motors script-follower script-leader detect-cameras detect-oak check-voltage check-voltage-follower check-voltage-leader check-oak photo calibrate-wb view-oak use-cuda use-cpu which-image view-camera record-vla train-smolvla train-pi0 push-checkpoint-vla eval-vla

include mk/setup.mk
include mk/robot.mk
include mk/dataset.mk
include mk/camera.mk
include mk/vla.mk
