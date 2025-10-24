PYTHON ?= python3
SAM ?= sam
VENV ?= .venv

ifeq ($(OS),Windows_NT)
VENV_BIN := $(VENV)/Scripts
else
VENV_BIN := $(VENV)/bin
endif

.PHONY: install clean test build deploy local

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt

test:
	$(VENV_BIN)/pytest -q

build:
	$(SAM) build

deploy:
	$(SAM) deploy

local:
	$(SAM) local start-api

clean:
	rm -rf $(VENV) .aws-sam
