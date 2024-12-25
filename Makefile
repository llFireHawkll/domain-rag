ENV_FILE := .env

clean:
	rm -rf dist/*
	rm -rf *.egg-info
	rm -rf build
	cleanpy -avf --exclude-envs .
	clear

install-package:
	make clean
	pip install .

install-all:
	make clean
	pip install -r requirements.txt
	pip install .

build-package:
	make clean
	python setup.py bdist_wheel

export_env:
	@export $(grep -v '^#' .env | xargs) && echo "Environment variables exported."