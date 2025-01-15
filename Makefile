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
	PYTHONPATH=. python src/train.py configs/config.yaml

.PHONY: formatters
formatters:
	PYTHONPATH=. black train.py src

.PHONY: lint
lint:
	PYTHONPATH=. nbstripout notebooks/*.ipynb
	PYTHONPATH=. tox
