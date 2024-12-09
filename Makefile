# Makefile for LaneWise Project (Windows-Compatible)

# Directories
SERVER_DIR=server
CLIENT_DIR=client

# Virtual environment directory
VENV_DIR=venv

# Commands
PYTHON=$(VENV_DIR)/Scripts/python.exe
PIP=$(VENV_DIR)/Scripts/pip.exe
NODE=node
NPM=npm

# Default target: Install dependencies
install:
	@echo "Creating virtual environment..."
	@python -m venv $(VENV_DIR)

	@echo "Activating virtual environment and upgrading pip..."
	@$(PYTHON) -m pip install --upgrade pip

	@echo "Installing backend dependencies..."
	@$(PIP) install -r $(SERVER_DIR)/requirements.txt

	@echo "Installing frontend dependencies..."
	@cd $(CLIENT_DIR) && $(NPM) install

# Run both backend and frontend servers
run:
	@echo "Starting backend server..."
	@$(PYTHON) -m uvicorn server.api:app --host 0.0.0.0 --port 8000 &

	@echo "Starting frontend server..."
	@cd $(CLIENT_DIR) && $(NPM) start

# Run tests for backend and frontend
test:
	@echo "Running backend tests..."
	@$(PYTHON) -m unittest discover -s $(SERVER_DIR)/tests

	@echo "Running frontend tests..."
	@cd $(CLIENT_DIR) && $(NPM) test -- --watchAll=false

# Clean build artifacts and generated files
clean:
	@echo "Cleaning frontend build..."
	@cd $(CLIENT_DIR) && $(NPM) run clean

	@echo "Removing backend models and photos..."
	@rm -rf $(SERVER_DIR)/models/*.joblib
	@rm -rf $(SERVER_DIR)/photos/*.png

# Display available targets
help:
	@echo "Available targets:"
	@echo "  install    - Install backend and frontend dependencies"
	@echo "  run        - Run both backend and frontend servers"
	@echo "  test       - Run tests for backend and frontend"
	@echo "  clean      - Clean build artifacts and generated files"
	@echo "  help       - Show this help message"
