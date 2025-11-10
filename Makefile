CAM_URL ?=
BASE_URL ?=
MODEL_DIR ?= $(CURDIR)/models/smolvlm2-mlx
CAR_API_STYLE ?= path
AUTOPILOT_LOG_LEVEL ?= INFO
PYTHONPATH ?= $(CURDIR)/..
EXTRA_ARGS ?=

.PHONY: check-env autopilot autopilot-debug

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
