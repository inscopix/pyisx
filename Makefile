repo=$(shell basename $(CURDIR))

.PHONY: test coverage-report jupyter 

jupyter:
	@echo "Installing kernel  $(repo) in jupyter"
	-yes | jupyter kernelspec uninstall $(repo)
	poetry run python -m ipykernel install --user --name $(repo)


install-poetry:
	@bash install-poetry.sh

install: install-poetry
	@echo "Installing py_isx..."
	poetry check --lock || poetry lock
	poetry install --verbose

install-test: install-poetry
	@echo "Installing py_isx & dependencies for testing..."
	poetry check --lock || poetry lock
	poetry install --extras "test" --verbose

test: install-test
	poetry run pytest -sx --failed-first

test-pip:
	@echo "Testing code installed on base env using pip..."
	pytest -s 


serve: install-poetry
	@echo "Serving docs locally..."
	poetry run mkdocs serve

setup.py: pyproject.toml README.md
	poetry run poetry2setup > setup.py


deploy: install-poetry 
	@echo "Deploying documentation to GitHub pages..."
	poetry run mkdocs build
	poetry run mkdocs gh-deploy