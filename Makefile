.PHONY: install data train test lint serve mlflow-ui docker-build docker-run

install:
	pip install -r requirements-dev.txt

data:
	python -m churn.data.generate

train:
	python -m churn.training.train

test:
	pytest

lint:
	ruff check .

serve:
	uvicorn churn.api.app:app --reload --port 8000

mlflow-ui:
	mlflow ui --port 5000

docker-build:
	docker build -t churn-mlops:latest .

docker-run:
	docker compose up --build
