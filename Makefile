CAM_URL ?=
BASE_URL ?=
MODEL_DIR ?= $(CURDIR)/models/smolvlm2-mlx
CAR_API_STYLE ?= path
AUTOPILOT_LOG_LEVEL ?= INFO
PYTHONPATH ?= $(CURDIR)/..
EXTRA_ARGS ?=
RASPI_AGENT_PORT ?= 8080
AGENT_URL ?= http://192.168.0.12:$(RASPI_AGENT_PORT)
DIRECT_INSTRUCTION ?= Approach the yellow box labeled TARGET and stop in front of it.
HYBRID_INSTRUCTION ?= Move toward the yellow TARGET box and stop exactly at the front.

.PHONY: check-env autopilot autopilot-debug raspi-agent pc-direct pc-hybrid

check-env:
	@if [ -z "$(CAM_URL)" ]; then echo "CAM_URL が未設定です"; exit 1; fi
	@if [ -z "$(BASE_URL)" ]; then echo "BASE_URL が未設定です"; exit 1; fi

autopilot: check-env
	PYTHONPATH=$(PYTHONPATH) python -m raspycar.autopilot \
		--camera-url "$(CAM_URL)" \
		--base-url "$(BASE_URL)" \
		--smol-model-id "$(MODEL_DIR)" \
		--car-api-style "$(CAR_API_STYLE)" \
		--log-level "$(AUTOPILOT_LOG_LEVEL)" \
		$(EXTRA_ARGS)

autopilot-debug: AUTOPILOT_LOG_LEVEL = DEBUG
autopilot-debug: autopilot

raspi-agent:
	PYTHONPATH=$(PYTHONPATH) python -m raspycar.raspi_agent \
		--bind 0.0.0.0 \
		--port $(RASPI_AGENT_PORT) \
		$(EXTRA_ARGS)

pc-direct:
	PYTHONPATH=$(PYTHONPATH) python pc_controller_direct.py \
		--agent-url "$(AGENT_URL)" \
		--instruction "$(DIRECT_INSTRUCTION)" \
		--model-id "$(MODEL_DIR)" \
		$(EXTRA_ARGS)

pc-hybrid:
	PYTHONPATH=$(PYTHONPATH) python pc_controller_hybrid.py \
		--agent-url "$(AGENT_URL)" \
		--instruction "$(HYBRID_INSTRUCTION)" \
		--model-id "$(MODEL_DIR)" \
		$(EXTRA_ARGS)
