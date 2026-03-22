# AcousticGuard Makefile — Common developer commands
# Usage: make <target>

.PHONY: help install download-data spectrograms train evaluate demo docker clean

# ── Colours ───────────────────────────────────────────────────────────────────
CYAN  := \033[96m
RESET := \033[0m
BOLD  := \033[1m

help:
	@echo ""
	@echo "$(BOLD)  AcousticGuard — Available Commands$(RESET)"
	@echo "  ────────────────────────────────────────────────"
	@echo "  $(CYAN)make install$(RESET)        Install Python dependencies"
	@echo "  $(CYAN)make download-data$(RESET)  Download ESC-50 benchmark dataset"
	@echo "  $(CYAN)make spectrograms$(RESET)   Pre-compute Mel spectrograms from audio"
	@echo "  $(CYAN)make train$(RESET)          Train the autoencoder (GPU auto-detected)"
	@echo "  $(CYAN)make evaluate$(RESET)       Evaluate on test set and show metrics"
	@echo "  $(CYAN)make demo$(RESET)           Open the live browser demo"
	@echo "  $(CYAN)make docker$(RESET)         Build and run the Docker container"
	@echo "  $(CYAN)make clean$(RESET)          Remove generated files and caches"
	@echo ""

install:
	pip install -r requirements.txt

download-data:
	@echo "$(CYAN)Downloading ESC-50 dataset...$(RESET)"
	@mkdir -p data
	@curl -L https://github.com/karolpiczak/ESC-50/archive/master.zip -o data/esc50.zip
	@cd data && unzip -q esc50.zip && mv ESC-50-master esc50 && rm esc50.zip
	@echo "Dataset extracted to data/esc50/"

spectrograms:
	@echo "$(CYAN)Generating Mel spectrograms...$(RESET)"
	python scripts/generate_spectrograms.py \
		--csv_path data/esc50/meta/esc50.csv \
		--audio_dir data/esc50/audio \
		--output_dir spectrograms
	@echo "Done. Spectrograms saved to spectrograms/"

train:
	@echo "$(CYAN)Starting training...$(RESET)"
	python src/train.py --config configs/default.yaml

evaluate:
	@echo "$(CYAN)Running evaluation...$(RESET)"
	python src/inference.py \
		--normal_dir spectrograms/engine \
		--anomaly_dir spectrograms/footsteps

demo:
	@echo "$(CYAN)Opening demo in browser...$(RESET)"
	@python -m webbrowser demo/index.html || open demo/index.html || xdg-open demo/index.html

docker:
	docker build -t acousticguard -f docker/Dockerfile .
	docker run --rm -p 8080:8080 -v $(PWD)/data:/data acousticguard

clean:
	@rm -rf spectrograms/ checkpoints/ __pycache__/ src/__pycache__/
	@find . -name "*.pyc" -delete
	@echo "Cleaned."
