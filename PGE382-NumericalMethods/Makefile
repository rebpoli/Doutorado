SHELL := /bin/bash

VENV := $(HOME)/.local/python_venv/Geomec
PIP := $(VENV)/bin/pip3
PY3 := $(VENV)/bin/python3
JUP := $(VENV)/bin/jupyter


venv:
	python3 -m venv $(VENV)

install: venv 
	source $(VENV)/bin/activate && $(PIP) install packaging jupyter pandas numpy matplotlib openpyxl scipy

notebook:
	source $(VENV)/bin/activate && $(JUP) notebook

