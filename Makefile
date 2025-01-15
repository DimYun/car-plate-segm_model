.PHONY:*

PYENV=/home/dmitriy/.pyenv/versions/3.9.17/bin/python
VENV=/opt/python_venvs/car_plate
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip

.PHONY: venv
venv:
	$(PYENV) -m venv $(VENV)
	@echo 'Path to Python executable $(shell pwd)/$(PYTHON)'

.PHONY: install
install: venv
	@echo "=== Installing common dependencies ==="
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: train
train:
	PYTHONPATH=. $(PYTHON) src/train.py configs/config.yaml

.PHONY: lint
lint:
	PYTHONPATH=. $(VENV)/bin/tox -e flake8,pylint

.PHONY: format
format:
	PYTHONPATH=. $(VENV)/bin/tox -e formatting

