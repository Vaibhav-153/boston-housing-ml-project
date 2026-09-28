.PHONY: setup data train lint test check run

setup:
	python -m pip install -r requirements-dev.txt

data:
	python scripts/download_data.py

train:
	python -m src.train

lint:
	ruff check .

test:
	python -m pytest -q

check: lint test

run:
	python app.py
