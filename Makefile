.PHONY: install test lint demo model investigate figures
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
investigate:
	python scripts/run_investigation.py
figures:
	python scripts/make_figures.py
