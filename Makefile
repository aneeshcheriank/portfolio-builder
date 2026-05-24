.PHONY: install test format lint all

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest
	# --cov specify the root folder for 

format:
	black *.py src/*.py tests/*.py main.py

lint:
	pylint --disable=R,C *.py

refactor: format lint

all: install format lint test