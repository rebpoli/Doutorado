SHELL := /bin/bash

_pyenv:
	python3 -m venv _pyenv
install: _pyenv
	source _pyenv/bin/activate && pip3 install packaging jupyter pandas numpy matplotlib openpyxl scipy

activate:
	source _pyenv/bin/activate

notebook:
	source _pyenv/bin/activate && jupyter notebook

