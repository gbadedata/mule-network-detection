.PHONY: install test lint demo model
install:
	pip install -e ".[dev]"
test:
	pytest -q
lint:
	ruff check .
demo:
	python run_demo.py
model:
	python run_model.py
