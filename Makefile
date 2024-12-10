# Makefile for LaneWise Project (Windows-Compatible)

# Directories
SERVER_DIR=server
CLIENT_DIR=client
VENV_DIR=venv

# Commands - Using forward slashes for MINGW64 compatibility
PYTHON=./$(VENV_DIR)/Scripts/python
PIP=./$(VENV_DIR)/Scripts/pip
NODE=node
NPM=npm

# Create required directories
create_dirs:
	@echo "Creating required directories..."
	@mkdir -p $(SERVER_DIR)/models
	@mkdir -p $(SERVER_DIR)/photos
	@mkdir -p $(SERVER_DIR)/data

# Default target: Install dependencies
install: create_dirs
	@echo "Creating virtual environment..."
	@python -m venv $(VENV_DIR)
	@echo "Activating virtual environment and upgrading pip..."
	@$(PYTHON) -m pip install --upgrade pip
	@echo "Installing backend dependencies..."
	@$(PIP) install -r $(SERVER_DIR)/requirements.txt
	@echo "Installing frontend dependencies..."
	@cd $(CLIENT_DIR) && $(NPM) clean-install

# Train the model
train_model: install
	@echo "Training ML model..."
	@cd $(SERVER_DIR) && ../$(PYTHON) lane_wise_system.py

# Run both backend and frontend servers
run: create_dirs install train_model
	@echo "Starting backend server..."
	@$(PYTHON) -m uvicorn server.api:app --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend server..."
	@cd $(CLIENT_DIR) && NODE_OPTIONS=--openssl-legacy-provider $(NPM) start

# Clean build artifacts and generated files
clean:
	@echo "Cleaning frontend build..."
	@echo "Removing backend models and photos..."
	@rm -rf $(SERVER_DIR)/models/*.joblib
	@rm -rf $(SERVER_DIR)/photos/*.png

# Display available targets
help:
	@echo "Available targets:"
	@echo "  install    - Install backend and frontend dependencies"
	@echo "  run        - Run both backend and frontend servers"
	@echo "  clean      - Clean build artifacts and generated files"
	@echo "  help       - Show this help message"