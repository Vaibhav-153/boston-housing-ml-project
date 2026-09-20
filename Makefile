.PHONY: setup data train test run

setup:
	python -m pip install -r requirements-dev.txt

data:
	python scripts/download_data.py

train:
	python -m src.train

test:
	pytest -q
	ruff check .

run:
	python app.py
