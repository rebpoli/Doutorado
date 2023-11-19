SHELL := /bin/bash

VENV := $(HOME)/.local/python_venv/_pyenv

_pyenv:
	python3 -m $(VENV)
install: _pyenv
	source $(VENV)/bin/activate && pip3 install packaging jupyter pandas numpy matplotlib openpyxl scipy

notebook:
	source $(VENV)/bin/activate && jupyter notebook

