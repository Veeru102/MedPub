# Makefile for MedPub project

# Variables
PYTHON = python3.11
VENV_DIR = backend/venv
BACKEND_DIR = backend
FRONTEND_DIR = frontend

.PHONY: all clean setup-backend setup-frontend run-backend run-frontend

all: setup-backend setup-frontend

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf $(BACKEND_DIR)/__pycache__
	rm -rf $(BACKEND_DIR)/**/__pycache__
	rm -rf $(FRONTEND_DIR)/node_modules
	@echo "Clean complete!"

setup-backend:
	@echo "Setting up backend..."
	rm -rf $(VENV_DIR)
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && \
		python -m pip install --upgrade pip setuptools wheel && \
		python -m pip install -r $(BACKEND_DIR)/requirements.txt && \
		python $(BACKEND_DIR)/download_nltk.py
	@echo "Backend setup complete!"

setup-frontend:
	@echo "Setting up frontend..."
	cd $(FRONTEND_DIR) && npm install
	@echo "Frontend setup complete!"

run-backend:
	@echo "Starting backend server..."
	. $(VENV_DIR)/bin/activate && \
		cd $(BACKEND_DIR) && \
		python -m uvicorn main:app --reload

run-frontend:
	@echo "Starting frontend development server..."
	cd $(FRONTEND_DIR) && npm run dev

# Helper targets
setup: setup-backend setup-frontend
	@echo "Setup complete! Use 'make run-backend' and 'make run-frontend' to start the servers."

run: run-backend run-frontend

.DEFAULT_GOAL := setup 