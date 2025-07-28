.PHONY: build run test clean shell push help

# Docker image name
IMAGE_NAME = pdf-heading-extractor
IMAGE_TAG = latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

# Default directories
INPUT_DIR = $(PWD)/input
OUTPUT_DIR = $(PWD)/output

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the Docker image
	@echo "Building Docker image..."
	docker build --platform linux/amd64 -t $(FULL_IMAGE) .
	@echo "Build complete. Image size:"
	@docker images $(FULL_IMAGE) --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"

run: ## Run the container to process PDFs
	@echo "Processing PDFs from $(INPUT_DIR)..."
	@mkdir -p $(OUTPUT_DIR)
	docker run --rm \
		-v $(INPUT_DIR):/app/input:ro \
		-v $(OUTPUT_DIR):/app/output \
		--network none \
		$(FULL_IMAGE)

test: ## Run tests with sample PDFs
	@echo "Running tests..."
	@mkdir -p test