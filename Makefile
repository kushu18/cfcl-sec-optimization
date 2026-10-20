.PHONY: install test test-cov api dashboard docker-build docker-up clean format lint

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	pytest tests/ -v

test-cov:
	pytest --cov=src --cov-report=term-missing --cov-report=html tests/

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
