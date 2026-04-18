# This Makefile automates the MLOps pipeline commands

# Variables
APP_NAME=mlops-portfolio
PORT=8000

install:
	pip install -r requirements.txt

train:
	python train.py

serve:
	uvicorn app:app --host 127.0.0.1 --port $(PORT)

monitor:
	python generate_report.py

test:
	pytest tests/

# --- Docker Commands ---
docker-build:
	docker build -t $(APP_NAME) .

docker-run:
	docker run -p $(PORT):$(PORT) --env-file .env $(APP_NAME)

# --- Standard Pipeline ---
all: install train test monitor
